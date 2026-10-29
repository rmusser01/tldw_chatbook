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

**A brand-new profile has no Browse section yet.** While every Library
source is empty the rail is three rows — **Import…**, **New note**,
**Explore all tools** — so there is no **Notes** row to click, and the route
above has nothing to land on. Three ways in from there: **Ctrl+N** (or the
bare **n**) makes a blank note and opens its editor; the rail's **New note**
row opens the New note view; and **Explore all tools** expands the rail to
its full **Browse** section, whose **Notes (0)** row opens the empty notes
list. The Browse row then arrives on its own — the rail graduates the moment
any one Library source holds something, and does not go back.

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
files file. Folder files' compact **‹ Files** action is not a reset. The
back cue reads "‹ " plus where it goes, one spelling across every Notes
and Folder-files surface.

On a wide terminal the strip above the canvas keeps both source switches —
**Library notes | Folder files** — whether Library navigation is open or
closed, and the in-canvas **‹ Notes** control is the way back to the list.
The **‹ Library / Notes** cue belongs to compact terminals, where the strip
collapses to it — see the Source strip bullet below.

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

When a resize crosses 120 columns, Notes puts focus back on the control you
were using once the new layout settles, and a scrolled pane such as the
Preview comes back at the same place, even when it lays out late on a busy
machine. A click or key press after the resize
wins: the control you clicked or moved to keeps focus, and the settled restore
leaves it alone. This holds in Library notes and in Folder files.

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
  otherwise be identical carry it. Opening such a note keeps that mark on a
  wide terminal: the editor's heading reads "Reading list · #0f3a" — the
  same third part its row shows — in Edit, Preview and Info alike, so which
  of the two you have open stays answerable after the list is out of view.
  **On a compact terminal it does not survive the heading strip.** At 100x30
  that strip carries the back cue and the source name either side of the
  title, and the title ellipsizes before them: the heading reads
  "‹ Back to list    Reading … Library notes" and the tie-break is gone, in
  Edit and Preview alike. The list row still carries it, so the answer is
  one **‹ Back to list** away rather than on screen. (Widening the heading's
  title budget is tracked separately.)
  While no note is
  open the list widens into most of the width the empty work area would
  otherwise waste, so long titles are not truncated on a wide terminal —
  at 235 columns the empty work pane still keeps a floor around 48-54
  columns for its own "Select a note to edit it here." message, so the
  list does not take the whole canvas; opening a note hands the rest of
  its width back. The same rule holds below 64 columns, where there is no
  room for two panes at all: with nothing open the list is the whole stage
  instead of sharing it with an empty work area, and opening a note gives
  the stage to the note. Reopening the list with its grip while a note is open
  (up to 82 columns) gives the list the whole stage rather than leaving
  the note a sliver beside it; the note stays open behind the list, and
  closing the list returns to the note with focus in it. Its own grip collapses or restores the list without
  changing the Folder files tree choice. When a rename SAVES, the note's list
  row catches up immediately, even while the title field still has focus — the
  row beside the editor is the only durable receipt on screen, so it is not
  allowed to contradict the heading and the save status beside it. The refresh
  the title field is protected from is a different one: a Notes refresh that
  lands mid-sentence never rebuilds the field under your hands, so the editor
  is not recomposed while you are typing in it and a body-only autosave costs
  the list nothing.

  Only one pane carries a "Next:" instruction at a time. With a note open, the
  work pane owns it; the list's own "Next: Create a note or add from files."
  is advice for a reader with nothing open and stands down until there is.

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
  scroll positions; on compact terminals use `‹ Back to list`. A key or
  click before the list has finished refreshing (`/` for the filter, say)
  keeps focus where it took you rather than pulling it back to that row.
  While you are
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
  only **‹ Notes**. With Import once, a Markdown file's YAML front matter is
  read and removed from the body when every property in it is one Chatbook
  reads (`title`, `tags`, `keywords`, `aliases`), or when the block is one
  Chatbook exported — it names a `note_id` (**Export Markdown**), or an `id`
  with `created_at` and `updated_at` (a bundle), so its `date` and ids are the
  export's own — and a note you exported comes back as it left; any other
  block, a daily note's `date:` beside its `tags:` included, stays in the body
  exactly as written (inside an Obsidian vault with the toggle on, see
  "Obsidian vaults" below). Keep a folder synced reads front matter only
  inside a vault and always keeps the block in the body.
- **Manage sync folders** — appears when roots or migration candidates exist;
  it shows text-explicit status and the valid action for each root. Escape or
  its **‹ Notes** returns to the Notes list the way leaving Add from files
  does: no note open, focus on the first note row, or on **New** when the
  list has no note row (no notes yet, or a kept filter that matches nothing).

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
snapshot. The tree scrolls at every terminal width — with the mouse wheel,
and as Down/Up or Tab move focus past the edge of the visible rows, which
scrolls the focused row into view one row at a time — so every folder,
note, "More notes" control and the "Recently deleted" row can be reached.
Each level loads independently in fixed pages of 20:

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
| "Filter notes… (Enter)" | Type words in any order and press Enter; the last word may be the start of a word (`Lisb` finds "Trip planning: Lisbon"); every word must appear in the title, the body or one of the note's keywords; wrap words in double quotes to match them as an exact phrase. The status line then reads "filter: \<text\> · N results"; with no match it says "No notes match “\<text\>” in titles, bodies or keywords. Try fewer words, or clear the filter." Enter again on the same filter opens its first result, and ↓ moves focus to the first result row (text you have not applied yet is applied first, as Enter would). Escape leaves the box and keeps what you typed in it; the filter that is applied does not change until you press Enter or **Clear filter**. |
| "New" | Opens the **New note** view — the same destination as the rail's **New note** row: **Blank note**, or **From a template…**. (**Ctrl+N** skips the view and makes the blank note itself.) Disabled while another notes operation is running. |
| "New folder" | Creates a folder in the tree beneath the toolbar. Disabled, with the reason in its tooltip, when the selected folder is sync-managed ("This folder is managed by sync; change its sync root instead.") or its branch is stale ("This branch may be out of date; retry it before changing it."). |
| Folder selected: "Rename" / "Move" / "Remove" | Act on the selected folder. Same two disabled reasons as **New folder**. **Move** opens 'Move folder "\<name\>" to…' with **Top level** first; the folder itself and everything beneath it are left out. After a move the destination opens with its own notes listed as well as the moved folder. If the folder changed somewhere else since the tree loaded, nothing is moved and the notice under the toolbar says so ("This folder was changed elsewhere since the tree loaded.") with a **Refresh** button that reloads the tree. |
| Note selected: "Add to folder" / "Move note" / "Remove placement" | File the selected note into a folder, move its placement, or take it out again. The picker is titled 'Move "\<title\>" to…' or 'Add "\<title\>" to folder…'. The picker lists every active folder at any depth; type to narrow it, 20 at a time with the exact total ("5 folders", "Showing 1–20 of 37 folders", or "No folder matches “\<text\>”"), with Previous / Next when there are more. The filter has focus when it opens; **Down** moves to the first folder, **Up** / **Down** walk the list, and **Enter** on a folder chooses it. **Choose** confirms the folder last focused and does nothing until you have focused one; **Escape** or **Cancel** closes the picker without changing anything. A sync-managed placement is refused with "This placement is managed by sync; change its sync root instead."; a note sitting in the automatic **Unfiled** group cannot have its placement removed ("Unfiled is shown automatically; move the note into a folder."). A note moved out of **Unfiled** leaves that group straight away. If the tree is out of date, the notice says what changed (for example "This note's placement changed elsewhere since the tree loaded.", or "The notes database was busy — nothing was written.") and offers **Refresh**, which reloads the tree; the next successful folder action replaces the notice. After you delete the selected note nothing stays selected, so these three actions are disabled until you choose another note. |
| "Restore folder" | Appears after a folder removal, to put it back. |
| "Sort: Newest" | Opens a one-row strip of Newest / Oldest / Title (✓ on the active one) in place of the action row; pick one directly, or press Escape to cancel. **Newest is the default.** The value is the order the folder tree is paged in, so choosing a new one reloads the tree (open folders included). While a filter is showing the control cannot own the order, so it renders blocked ("○ Sort: Newest") and the line under the toolbar says why: "Sort unavailable — clear the filter" (task-32549). |
| "Add from files…" | Choose **Import once** or **Keep a folder synced** before selecting a source. |
| "Manage sync folders" | Appears only when roots or paused migration candidates exist; opens root status and contextual controls. |
| "Last import" | Reopens the latest import receipt from this app session after you return to the Notes list. |
| "Export…" | Opens the "Export bundle (.zip)" canvas scoped to notes — bundle notes into a .zip. |
| "Select" / "Done" | Toggles select mode: rows grow ☑/☐ checkboxes, and a row appears with "N selected", "Select all N shown", "Clear", "Export selected" and "Actions…". **Space** checks or unchecks the focused row, as Enter does. "Select all N shown" selects exactly the N notes on screen — the rows the tree has loaded — so the number it names is the number it selects, with a filter applied or after coming back from Export. The count is also repeated on its own line below the row, and the two always read the same number. "Export…" hides while selecting. The row fits the list pane it sits in, not the terminal: when the pane cannot hold it on one line, Export selected and Actions… move to a second row; when even that does not fit (a compact terminal, or the narrower list pane a 120-column terminal gives) the row shortens to "Done", "All N", "Clear", "Export" and "Actions…" and drops its own copy of the count, keeping the line below it — every action stays on the pane. With nothing checked, "Export selected" and "Actions…" are blocked, and the count line under the row names it: "0 selected — Export selected unavailable" (task-32549). The reason is on that line rather than in the button, because this row is measured to the cell and a line of its own would cost the list a row; it reverts to the plain count the moment you check something. "Select all" and "Clear" keep keyboard focus after you press them, so `e` exports straight after "Select all". **Actions…** opens a strip of Move to folder… / Add keyword… / Remove keyword… under the row (Escape closes it and returns to Actions…). Move to folder… opens the same all-folders picker as Move note, titled "Move N notes to…", and puts every checked note in the folder you choose, taking it out of the folders it was filed in by hand (a placement a sync root manages stays). Add keyword… and Remove keyword… ask for one keyword and add it to, or remove it from, every checked note; text with no keyword in it (a lone comma, say) changes nothing and says so: "No keywords entered — nothing changed." Nothing changes until you choose a folder or press Apply; Escape or Cancel leaves every note as it was. A receipt then names the count — "Moved 3 of 3 notes to Projects / Thesis", "Added 'travel' to 2 notes · 1 already had it", "Removed 'travel' from 3 notes" — and select mode ends. If some notes could not be changed the receipt says how many, select mode stays, and only those notes stay checked, so the same action retries just them. Notes cannot be deleted in bulk. A note open beside the list turns into a labelled read-only preview for as long as select mode lasts — its header reads "Read-only preview · Included / Not included in bulk selection", and Save, Delete, "Use in Console", Copy and the exports are disabled — so a bulk action cannot be issued from a pane that still looks editable. Pressing "Done" hands the editor straight back. Escape, or applying or clearing a filter, ends select mode the same way and clears the checks. |

**Keyboard on this toolbar (the Notes list only).** `/` focuses the filter
and selects whatever is already in it, so the next thing you type replaces
the old query rather than landing in front of it — no need to clear the
field first. (Once the filter
has focus, `/` is an ordinary character; see the **/** row under "Keyboard &
commands".) From the filter, Tab walks the toolbar in the order it is drawn,
and a disabled control is skipped — so the count to any one button depends
on which of them are live, not on the button's position. A filter is one
cause (it disables **Sort**); the other two are an empty list, which
disables **Select** ("○ Select" until the first note exists), and a notes
operation in flight, which disables **New**, **Sort** and **Select**
together. The two counts below assume a list with notes in it and nothing
running: **Select** is also disabled at zero notes, which moves the same
counts.

- **Unfiltered**, Tab ×1–4 from the filter reaches **New**, **Sort**,
  **Select**, **Add from files…**.
- **Filtered**, Tab ×1–4 reaches **New**, **Select**, **Add from files…**,
  **Export**.

Read the highlighted control before pressing Enter; the count is not the
same in both states.

With no notes at all, the list reads "No notes yet. Create your first note."
above the tree — even when the seeded **Agent_Lessons** folder (see "Reuse
solutions with Agent Lessons" above) is the only row showing, so a
first-time user is never left staring at one unexplained folder with no
other cue. While the library holds zero notes, that folder row itself also
carries a one-line gloss: "Agent_Lessons — where Console agents file
reusable lessons (empty)".

On a terminal narrower than 64 columns the list pane is narrow enough that
the status line and the toolbar's third action would otherwise paint
mid-word ("…or add from" instead of "files.", "Sel" instead of the full
action name). At that width — and only there; at 64 columns and up the line
keeps its full wording — the status line drops its "Library notes ·"
prefix, since the source strip above it already says which notes these
are, and the toolbar moves the action that does not fit onto a row of its
own. Nothing is ever painted as half a word.

At other compact widths, the list's introductory status wraps to the height
it needs: at 80 columns, the complete next action occupies three rows.

This rule holds for every toolbar row at every width, not only the browse
actions on a narrow terminal (task-32544, task-32557). Any group of
actions wider than the pane it is in wraps onto as many rows as it needs,
in reading order — which is what the folder actions do beside an open note
on a wide terminal, where "New folder  Add to folder  Move note  Remove
placement" needs one more cell than the list pane has. A pane that
narrows re-shapes its toolbar at once, reading the pane's actual resolved
width rather than the one its last compose had. A
pane that *widens* keeps its shape until the next time the list refreshes —
a row with room to spare costs nothing, and re-shaping on every resize would
undo the in-place updates that keep a terminal drag cheap.

The same wrapping applies when no note is selected: on the wide layout,
disabled folder actions keep their complete labels and the selection explanation
stays below them. The compact layout keeps its existing single **New folder**
action so the list retains its row budget. On the wide layout these actions sit
under a "Folders & placement" heading, and with no note selected the
explanation reads "Note actions unavailable — select a note in the list".

### Edit, Preview, and Info

| Control | What it does |
|---|---|
| "‹ Notes" / "‹ Back to list" | Returns to the list (your text is already saved — see autosave below). One wording across Edit, Preview, and Info: "‹ Notes" at wide sizes, "‹ Back to list" on a compact terminal. |
| Where this note lives | One row under the title, on a terminal 80 columns or wider. A note that exists only in the Library database reads "In the Library database only — no file on disk". A note that lasting sync keeps in step with a file reads "In a synced folder", then that file's path, then when the file was last written — the file's own time, so an edit made in Obsidian counts too. The path is elided in the middle when the pane is narrower than the row, keeping the file name; the write time is dropped before the path is, and the first part is never shortened. Both facts are read fresh when the note opens, from the sync relationship and from the file itself rather than from anything stored on the note. The time is the file's last write: a save schedules the write and the folder pass that performs it runs a moment later, so the row re-reads itself once that pass has landed (it waits up to five seconds for it) rather than on the save; reopen the note if it still names the previous write. That is the row being accurate rather than reassuring — it is telling you whether the file has been written (task-32640). When the note's sync folder is held for attention — an open sync entry, a conflict waiting for review, a failed pass — the row says so right after the world, "In a synced folder · ⚠ Sync needs attention · …", and the status line above the editor reads "Saved HH:MM in Notes · ⚠ Sync needs attention · Next: Open Manage sync folders." instead of a plain "Saved": your text is in Notes, but nothing reaches the file until the folder is resolved (TASK-34000.2). |
| **Edit** | Shows the editable title, keywords and body. This is the default view when you open a note. Keywords sit on their own row under the title — comma-separated, edited in place, and reachable with one Tab from the title, so you no longer have to open **Info** to add one. It is the same shape as the Title field, which costs the body five rows at a wide size and one on a compact terminal. The same field appears in Info's Properties; whichever you type in, it is the same keywords. |
| **Preview** | Shows the note's title above the body, rendered as Markdown, without replacing your draft. It takes the whole work pane, and it takes keyboard focus when you open it, so `pgup`/`pgdn` page the rendered note straight away — no click inside the box first. If the body's first line is an H1 that exactly repeats the note's title (`# ` and the same words — the shape most exported Markdown files have), Preview shows it once, as the title line, instead of printing it twice. A rendered heading is left-aligned, where the rest of the body begins, rather than centred like a page banner — so a note whose title came from Obsidian frontmatter, which never matches its first heading exactly, shows a title and a heading rather than two titles. An Obsidian callout renders as a quoted block headed by its type, on its own line above the body — `> [!note] Title` becomes "Note: Title", `> [!warning]` on its own becomes "Warning" — rather than printing its `[!note]` marker or running the type into the first line of the callout; a callout written without the space (`>[!note]`), a folded one (`> [!note]-` / `+`), a nested one (`> > [!tip]`) and a capitalised type (`[!TODO]`) all render the same way, and an example inside a fenced code block is left exactly as you wrote it. Tab moves through the same controls Edit offers, and the footer names each one as you reach it. Escape leaves Preview for the **list**, not back to Edit — the footer says so, "esc back to notes" on a wide terminal and "esc notes" on a compact one (the same key, the same destination, shortened to fit). The status line does not offer to keep editing while Preview is showing; it names **Edit** instead. Clicking a note link in Preview opens that note; with the Preview focused, Enter follows the note's link or lists its links to choose from. The note you reach shows **‹ \<title\>** in its heading, which takes you back to the note you came from. Web links open in your browser; nothing else is handed to the operating system. |
| **Info** | Shows Properties (including comma-separated keywords, note dates/version, and **Linked from**), Reuse & Export, and Danger sections. Each property is on its own row, labelled — Created, Modified, Version, Words — rather than joined into one sentence: on a wide terminal the values line up in a second column, and on a compact one the rows keep a single column so a timestamp is never cut off the pane. |
| **Linked from (N)** (Info → Properties) | Lists the notes whose bodies link to this one, newest import or not — the `[[target\|title]](note://…)` links Import once writes for an Obsidian vault's `[[wikilinks]]` (see "Obsidian vaults"). Click an entry to open that note; its heading then shows **‹ \<title\>**, which returns to the note you came from — one hop, so the control is gone once you are back. While the lookup runs the line reads "Linked from — checking…", and "Linked from — couldn't check" if it failed, so a count is only claimed once the answer is in. When nothing points here the line reads "Linked from (0) — no notes link here yet". A linking note you delete drops out of the list and returns when it is restored. The list is capped at 50 entries; past that the count reads "50+". Type `[[` in the body to pick a note: the picker narrows as you type, Enter inserts the link, Esc leaves your text alone. Resizing the terminal closes the picker without inserting anything — keep typing to open it again. Only typing opens the picker: moving the caret with Home, the arrow keys or a click into a link you already have never opens it, so navigation keys alone can never change which note a link points to. A `[[Title]]` you type by hand counts too when exactly one note has that title (case is ignored for A–Z letters, so `[[weekly]]` finds "Weekly", but an accented letter must match exactly — `[[élan]]` does not find "Élan"); a title no note has, or two notes share, is left as plain text. Links are recorded when the note is saved, so a note written before this version picks up its typed `[[Title]]` links the next time it is saved. The answer is looked up rather than searched for: each note records the links its body carries when it is saved or imported, so the lookup costs what a note's own inbound links cost rather than growing with the size of your vault. The stored `[[title]](note://…)` links an existing library already holds are picked up the first time this version opens it — nothing to re-import; only the hand-typed `[[Title]]` form waits for the note's next save. |
| Status line | Shows the autosave state: "Saved", "Saving…", "Unsaved changes", "Conflict — …", "Save failed — …", or "Unavailable — …". Its "Next: Keep editing; changes save automatically." appears only while a save is counting down, running, or paused behind an open dialog (or the note is clean, so your next keystroke starts one); a note with unsaved changes and no save pending reads "Next: Keep editing; press Save to save now." instead. Right after a save completes in this session the state names the time — "Saved 12:47" — **in your local time**, the same clock Info → Properties prints, so the two never disagree about when that save happened; leaving the note and reopening it later in the same session goes back to the bare "Saved". It does not carry a word count. Created/Modified/version details are under Info → Properties, one labelled row each, with an absolute local timestamp beside its relative age and the word count (e.g. a "Created" row reading "2026-09-08 21:14 · 3m ago", then "Modified", "Version" and "Words"). In Info, "Saved" appears once. |
| Chrome strip | One row directly under the body, right-aligned: "N words · L:C" — the words in the note and the caret's line and column, both counted from 1. The count is the whole open note's, at any length: a 37 KB, 5,427-word note reads "5,427 words · 1:1" the moment it opens, whatever you had open before it, and the thousands separator is part of the number. It follows your typing and your arrow keys with no save and no reload. Nothing on it is a control; Tab never stops there. It appears while **Edit** is the open view on a terminal 80 columns or wider; **Preview** and **Info** have no caret, and a narrower terminal gives the row back to the body. The strip does not repeat the save state — that stays on the status line above the mode controls. The editor's Created/Modified/version line lives in one place, Info → Properties. |
| **Save** | Saves immediately, without waiting for autosave. It remains visible at every terminal size: beside the mode controls when the editor pane is wide enough for the whole header on one row, otherwise on the row directly under Edit / Preview / Info, with the save state on its own line above them. The header never pushes Save past the pane's edge, and F6 only lands on a Save you can see. |
| **Use in Console** | Hands the note to the Console as staged context, with the suggested prompt "Use this note as context and help me work with it." It remains visible beside **Save** at every terminal size, with its whole label — never a clipped "Use in". On a new note you have not typed into, **Discard new note** sits beside it (worded "Discard" only where its row is too narrow for the full label — a 120-column terminal; wider stacked rows keep "Discard new note"); it keeps its place on the row when it does not apply, so Edit / Preview / Info never shift. |
| **Copy** (Info) | Copies the note to the clipboard as Markdown — "Note copied to clipboard as markdown!" |
| **Export Markdown** / **Export text** (Info) | Saves the note to a file you pick. The picker opens in the folder you last exported to this session (your home folder until then). If a file of that name already exists there, a "Replace existing file?" prompt names the file and its folder — `Replace "Reading list.md" in ~/exp?` — with **Cancel** focused; Escape, a click outside the prompt or **Cancel** leave the existing file exactly as it was, and only **Replace** writes. If the file you chose is a link to another file, the prompt names that file and its folder, and Replace writes through the link, leaving the link in place. The write is atomic (a temporary file in the same folder, then a rename), so a failed write never leaves a truncated file. Success shows "Note exported successfully to ~/exp/Reading list.md" — the folder and the file, not just the name. Closing the file picker without choosing a file (Esc or Cancel), or keeping the existing file at the replace prompt, is not a failure: the only sign is the notification "Note export cancelled." (naming the kept file), and the status line keeps the note's own state. After an export — cancelled, written or failed — and after **Copy**, keyboard focus is back on the button you used, so Enter repeats it — never on **Delete**. |
| **Delete** (Info → Danger) | Asks inline, in place — Info stays open and the prompt renders inside the Info box, on the row directly under the Delete button that raised it, at every terminal size: "Delete this note? Undo will be available in the Notes list." Info scrolls to the prompt by itself, so its copy and both buttons are in view without a manual scroll, and stay in view if the terminal is resized while it is open. Tab / Shift+Tab cycle only between **Cancel** and **Delete** while it is open, and the footer names whichever one is focused ("enter cancel" or "enter delete"). Every other Info action — including "‹ Notes" / "‹ Back to list" — is disabled until you choose Cancel or Delete. Cancelling puts focus back on Delete and leaves Info scrolled exactly where it was. A successful delete returns to the list with a named "✓ deleted · …" receipt offering **Undo** and **Dismiss**. |

Opening a note shows "Loading note…" only while the note is being read. If a
read takes longer than about three seconds the editor stops waiting and shows
"Unable to load note — timed out after 3 s. Press Retry." with a **Retry**
button; "‹ Notes" / "‹ Back to list" (the same compact-vs-wide wording as
Edit/Preview/Info) takes you back to the list, and opening another note still
works.

**Autosave** runs about two seconds after you stop typing; the meta line
flips to "saving…" and back to "saved". If you keep typing without a
two-second pause, it still saves at least every ten seconds, so a long
burst of steady typing is never held back until you stop. A note in a synced
folder saves on the same timings: a save never waits for the folder's previous
sync, and a save that lands while that sync is still writing the file settles
on its own (see "Add from files and lasting sync" below: typing in a synced
note never holds its folder). Opening help (F1), a dialog such as Move note
or Add to folder, or the command palette pauses the countdown while it is
open — nothing saves behind a dialog — and the countdown resumes the moment
the dialog closes, so an edit typed just before F1 is still saved within
the normal delay after you close it. If an autosave is
refused — a title with a leading or trailing space, unsafe markup, a
duplicate keyword — or the write fails, the status line says why, and your
cursor stays where it is, whether that is the field you are typing in, the
Info pane or a button. Nothing jumps to another field or pane mid-sentence.
The field that needs fixing is marked, with the reason directly under it.
On a wide terminal the field is outlined. On a narrow one (under 120
columns) the Title and Keywords boxes have no outline, so their text is
shown in the error colour instead — a title that is only spaces has no
text to colour, and shows its mark as the reason row alone — while the
Body keeps its outline at every width. On the narrowest terminals that
one-row reason may be cut short at its end; the status line above still
carries all of it. The status line reads "Fix the marked field, or press
Save to go to it", and it keeps saying so (never "changes save
automatically") until you change the marked field; pressing **Save**
takes you there. Leaving the Title field
tidies a leading or trailing space, and leaving Keywords drops a duplicate
that differs only in case ("x, X" becomes "x") and an empty entry ("a,,b"
becomes "a, b"), so an ordinary slip like the duplicate never leaves the
note unsaved; a list with nothing to drop is left exactly as you typed it
("a,b" stays "a,b"). You see the edit in the field, and nothing is
rewritten at save. Switching to another window and back does not count as leaving the
field: what you typed stays exactly as it was. If the same note was changed somewhere else while you were editing,
a banner appears: "This note changed elsewhere — Overwrite saves your text;
Reload discards it." — pick **Overwrite** or **Reload**. Either way, once
the banner closes the keyboard is back in the note body.

**Quitting with unsaved changes.** **Ctrl+Q** saves the open note before
Chatbook exits, so text typed in the last moments is kept, and a new note
keeps the title and body you typed. If that save cannot finish — the title
is one the save refuses, the note changed elsewhere, or the write fails —
Chatbook does not exit. It asks 'Quit and discard unsaved changes to
"<title>"?' with **Keep editing** selected: Keep editing (or Enter, or
Escape) returns you to the editor with your text intact, and only **Discard
and quit** exits without saving it. The same question covers an unsaved
Prompt or Skill draft and a Folder files edit that could not be saved.

While any editor field — the title, the body, or either keyword box — has
keyboard focus, nothing repaints the editor underneath you: a refresh that
arrives mid-sentence — a save landing, the
first note reaching the list, the Library graduating to its full rail — leaves
the editor alone, so keystrokes never land in the wrong box and focus never
jumps away as you type or after you move to another field. The next refresh
that arrives once your hands are off the field paints normally.

A refresh of the Notes list also leaves alone a control outside the list
that you clicked or Tabbed to: a pane grip, or **Library notes** / **Folder
files** on the source strip, keeps focus when the list repaints beside it.

**Ctrl+S** saves the open note now, without waiting for the 2-second autosave,
from anywhere in the note's pane: the body, Title, Keywords, the
Edit/Preview/Info buttons or Info. It does what the **Save** button does, and
focus stays where it was. With nothing unsaved it says "Saved — no changes."
on the status line, with no pop-up — the same answer the **Save** button gives.
Where a save is held it says why on the status line instead, with no pop-up,
and moves nothing: "Save locked — finish or cancel the destructive action."
while the Delete confirmation is open, "Save locked — resolve the note
conflict below." in a conflict (the conflict choices are brought into view).
With focus on the Notes list, the filter or the folder tree, Ctrl+S does
nothing, even with a note open beside it. Autosave still handles ordinary
typing.

#### Editor keys

| Key | What it does |
|---|---|
| **Tab** / **Shift+Tab** | Move between the editor's own controls and stay there: "‹ Notes", Edit, Preview, Info, Save, Use in Console, Title, Keywords, Body, and back round to "‹ Notes". Tab out of the body stays inside that cycle, never wrapping round to the "Library notes / Folder files" switch above the pane. The move lands before the next keystroke, so typing straight through a Tab puts the rest where you meant it. |
| **F6** / **Shift+F6** | Leave the editor for the Notes list or the Library rail. This is the way out of the editor's Tab cycle; **Escape** is the other (it returns to the list). |
| **Ctrl+End** / **Ctrl+Home** | Jump the caret to the end or the start of the note body. `End` and `Home` still move within the current line. |
| **Ctrl+S** | Save the note now (see above). The footer and F1 show `ctrl+s save now` (`ctrl+s save` on a compact terminal) only while the note has unsaved changes — the key would save. |
| **Escape** | Returns to the list — one press, from Edit, Preview, or Info. From Info it goes back to the editor first. |

Arriving in the **Title** by keyboard puts the caret at the end of the
existing title; it does not select the title, so one keystroke can no longer
replace it. Select the text yourself (Shift+Home, or drag) when you do want
to overwrite. The keyword boxes behave the same way.

In Edit and Preview, the footer names whichever control has focus as an
"enter …" chip, so focus is never unaccounted for: Tab onto "‹ Notes" and
the footer reads "enter back to list", onto Save and it reads "enter save
note". While the body or a field has focus there is no enter chip, because
Enter types. Info names its focused control the same way (see below), and
while its inline delete confirmation is open the chip names the focused
button ("enter cancel" / "enter delete").

That is true in **Preview** and **Info** as well, not only in Edit: Tab there
walks the same six controls — "‹ Notes", Edit, Preview, Info, Save, Use in
Console — and the footer names each one beside the tier's own keys, so an
Enter pressed in a reading pane is never a guess. In Info that extends to the
pane's own actions: Copy, Export Markdown, Export text and Delete each name
themselves ("enter delete note"), and the Keywords field shows no enter chip
because Enter does nothing there.

In **Preview**, where the rendered body is the last stop of the cycle, Tab
out of it lands on **Edit**, the first of the mode buttons, not on "‹ Notes".
"‹ Notes" is one Shift+Tab back from Edit, and Escape still leaves the note;
the swap exists so that the one obvious Tab-then-Enter out of a reading pane
cannot close the note by accident. In **Info** the body is not the last stop
— Keywords, Copy, Export Markdown, Export text, Delete and "‹ Notes" follow
it — so Tab there simply carries on to Keywords and the cycle reaches every
one of the pane's own actions.

When the note body has keyboard focus, only its boundary becomes more
prominent. The body background and editor size stay unchanged, so focusing
the editor does not flash or fill the writing surface. Small fields such as
Title and Keywords still use their usual filled focus treatment. Preview's and
Info's reading panes switch to a heavy accent border while they have focus.

The editor's `‹ Notes` control, the compact `‹ Library / Notes` cue and
Escape all use the same guarded return. A dirty save, sync, conflict, reload confirmation,
or running mutation can therefore keep the focused task open until it is safe
to leave. A successful return restores the Library notes / Folder files source, filter,
sort, selected note or placement, Notes-list scroll, Library-rail scroll, and
semantic keyboard focus instead of starting over at the first row.

If a save is blocked — most commonly a title that starts or ends with a
space — Escape does not leave silently. The notification repeats the reason
the status line shows, which names the field and the fix: "Can't leave yet —
Title begins or ends with whitespace — remove it to save.", or "Can't leave
yet — Keywords contain a case-insensitive duplicate — remove one to save."
Clicking another destination in the nav bar while a save is blocked shows
the same reason with the destination named ("Can't open Console yet: Title
begins or ends with whitespace — remove it to save."), keeps Library
highlighted, and the next click on that destination goes through once the
field is fixed.

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
note remains deleted. See "Recently deleted" below, which also recovers a
note whose receipt was dismissed.

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
anything, or quit Chatbook, the blank note is quietly discarded rather than
left behind as a stray "Untitled" row. Escape, another rail row or another
tab discards it the same way.
Pressing "Save" keeps it, and so does typing anything **that is not only
whitespace** — a title of nothing but spaces, with an empty body and no
keywords, still counts as blank and is discarded on the way out, quitting
included (Chatbook does not ask about it). Because
you did type something there, that discard is not silent: leaving says
"Empty note discarded", and the note's row leaves the list at once rather
than lingering as an "Untitled · now" row for a note that is already gone.
(A note you never touched at all is still discarded without a message —
nothing was written to lose.) That includes naming it "Untitled"
yourself: once you have touched the title field the note is yours, and it
is kept even with an empty body. A note kept with an empty title is saved,
and listed, as "Untitled". **From a template…** unfolds eight template
rows that pre-fill title, body, and keywords instead; each row shows the
template name with the title the note will get. Available templates:
Brainstorming session, Bug report, Code review, Daily journal entry, Meeting
notes, Project planning, Research notes, Todo list. The rows stay folded
until you ask for them, and fold again the next time you open the view.

This view is the task in hand while it is open, and it gets the pane width
that goes with that (task-32544, task-32547). At 235 columns the empty list
keeps to its own comfort width (around 62), leaving the view itself about 105
columns alongside the roughly 37-column navigation rail, room enough for
the view's own status line to fit on one row.
On a terminal too narrow to show both at once — 60 columns, say — the view takes
the stage outright. Escape returns to the list at the same size.

Opening this view parks keyboard focus on **Blank note**, so Enter creates
a note straight away without tabbing to find it; ↑/↓ move between Blank
note, **From a template…** and (once it is open) the template rows, and the
focused row carries the same left-edge bar the Notes list rows use. The footer's "enter create note" appears only
while one of those rows genuinely has focus — move to "‹ Notes" / "‹ Back to
list" (the same compact-vs-wide wording as Edit/Preview/Info) and it drops,
because Enter there goes back rather than creating anything.

**Leaving the view.** Escape, the view's back control (**‹ Notes** on a wide
terminal, **‹ Back to list** on a compact one) or the rail's **Notes** row
returns to the Notes list with no note open — on a wide terminal the work
area beside it reads "Select a note to edit it here." — and puts focus on the
first note row, where opening Notes puts it, not in the filter, so the next
letter is a list key rather than filter text. When the list has no note row
— no notes yet ("No notes yet. Create your first note."), or a kept filter
that matches nothing — focus goes to its **New** button instead. (Opening
an empty Notes list from the rail, from anywhere other than Create, puts
focus in the filter.)

### Add from files and lasting sync

**Add from files…** first asks what relationship you want. Until you choose,
the header names neither relationship — one heading, "Add files to Library
notes.", over a status line that says nothing has been chosen yet, and its
next action is to pick one. It opens on that chooser every time, whatever the
previous Add from files or Manage sync folders step ended on; only a check or
activation that is still running is shown instead, until it settles.

Each option names the difference that matters most and is easiest to miss:
**Import once** reproduces your folder structure as Library folders, while
**Keep a folder synced** collects every note in one managed Library folder.
Below them, a third line points at the answer that is on neither button —
**Folder files**, the mode of this same screen that edits the folder where it
is and imports nothing; the strip above the canvas is where you switch to it.
The bar below the chooser holds only **‹ Notes**. Escape or **‹ Notes** —
from the chooser, from **Import once**, or from **Keep a folder synced**'s
setup and review — returns to the Notes list the way leaving the New note
view does: no note open, and focus on the first note row rather than in the
filter, or on **New** when the list has no note row (no notes yet, or a kept
filter that matches nothing). Until lasting sync is ready
on the profile, the second button reads "○ Keep a folder synced" with the line
"Keeping a folder synced isn't ready on this profile yet. Nearest valid
action: Import once." under it.

- **Import once** copies supported files into Library notes and ends after its
  reviewed receipt. Later changes to the originals are not tracked.
- **Keep a folder synced** creates a lasting local relationship. Choose the
  folder, direction, and local Library destination, then choose **Check
  changes**. Checking is mutation-free. Review safe actions, attention items,
  skips, filesystem effects, and deletion-like effects before **Activate
  reviewed root** is enabled. Each row reads the same way an Import once row
  does — the file's path, what will happen to it, and the Library folder it
  lands in ("Daily/2026-09-06.md · Create a Library note · PowerVault") —
  under a heading per effect carrying its count ("Create a Library note
  (56)"). The summary line is followed by the pass's own scope — "Syncs .md,
  .markdown and .txt only; other files are left alone" — because a folder's
  `.csv` and `.yaml` sources are ignored here rather than reported, which is
  the one place this path and Import once differ on the same folder: Import
  once lists them under Failed or Skipped with a reason. A folder whose files all get the same effect collapses to one
  summary row you can open ("▶ Archive · 45 files · Create a Library note ·
  PowerVault"). A run collapses at eight files; seven or fewer are listed one
  per row, skipped files from one folder follow the same rule, and Import once
  uses the same threshold. Files the check leaves alone appear under **Skipped (N)**
  with the reason on the row (".trash/Old idea.md · Obsidian trash —
  skipped", "Inbox/Untitled.md · Empty file — nothing to import").

Both folder pickers remember where you were. Each reopens at the directory it
last picked in *that* flow, so Import once and Keep a folder synced never move
each other's starting point, and neither borrows the Library ingest browser's.
The first use of either — or a remembered folder that has since been moved or
deleted — falls through to the folder `[notes] sync_directory` names, and only
then to your home directory. **Folder files** keeps its own separate memory,
see [File notes](file-notes.md). In all three Notes folder pickers, folders
list first, a folder row shows how many notes sit directly inside it
("12 notes") and marks an Obsidian vault "· vault", and **Ctrl+R** offers the
folders that picker chose before — see [File picker](../file-picker.md).

**What a folder has to be before it can be checked.** The folder itself must be
a real folder (not a link to one) on a local disk, outside Chatbook's own data
directory, not already connected as a sync folder, and not the Folder files
root. Every `.md` file inside it is checked too. A file the check can list but
cannot read as note text is listed in the review as a skipped row that names it
and says what to do — `notes.md · Mixed line endings — save it with one style`,
`latin1.md · Not UTF-8 text — convert it to UTF-8 or move it out of the folder`,
a file you cannot write or do not own, one with more than one name on disk, or
one that changed while it was being read. The rest of the folder syncs as
usual, and a skipped file is picked up by the next Check once it is fixed or
removed. A migrated folder cannot be activated while one of its own notes'
files is skipped; the review says which file and holds Activate until it is
fixed.

A few things still stop the whole folder before any file is read, because the
folder listing itself refuses them: a file larger than 10 MB, a `.md` that is a
link rather than a file, and an entry that is not an ordinary file (a pipe, a
device). **Check changes** refuses a folder that fails its own rules and names
which rule it was, in the setup pane's status line, with the next action — for
example "That folder is inside Chatbook's own data directory. Pick a folder
outside it, then Check again", "Another Chatbook window is using that folder",
"That folder is already connected", or "Some files there are larger than 10
MB". A refusal changes nothing: pick a different folder, or fix the cause, and
**Check changes** again in the same session.

If files or notes change after checking, activation is refused as stale and the
nearest valid action is **Check again**. Conflicts and deletion choices are not
silently settled by a global winner policy. Server setup is visibly disabled
with **Unavailable — server sync-folder capability not installed**.

For an eligible conflict where the same bound note and file both changed,
choose **View comparison** to inspect their current text and metadata, then
stage exactly one choice:

- **Keep file** updates the bound Library note from that file.
- **Keep note** replaces that file with the bound Library note.
- **Keep both** first preserves the original Library-note text as a new,
  unbound manual note, then updates the original bound note from the file.
- **Skip for now** changes nothing. The conflict stays in **Needs attention**
  and can be reviewed later.

Staging or changing a choice does not alter either authority. Choices you
stage in a review are kept while the folder is unchanged, even if you leave and
come back: the status line then reads "Choice staged. No changes yet." (or "2
choices staged. No changes yet."), and nothing is applied until you choose
**Apply reviewed**. A folder that changed in the meantime gets a fresh review
with no choices staged. A review opened from **Manage sync folders** offers
**‹ Sync folders** to go back there, and Escape goes there too (the footer
reads `esc back to sync folders`); its receipt offers **‹ Sync folders** alone,
since **Manage sync folders** would open the same list. One reached from
**Add from files…** offers **‹ Notes**, and Escape returns to Notes. **Apply
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
Before a new root is activated, its review shows "Resolution history
unavailable — it starts after this root is activated" on the line above the
action bar.

Deletion, identity, move, representation, duplicate-authority, managed-folder,
pause, capability, and activation attention remain blocked; this conflict
review does not turn them into content choices. Unsupported filesystem writes
fail closed, leave the root in attention or recovery, and do not escape the
configured sync folder.

**Manage sync folders** lists active, paused, passive, offline, attention,
recovery, stopped, and migrated-candidate states. Use **Check changes** to scan
an available root. A check that finds effects takes you straight to them: the
status line reads "Manual check finished. N change(s) to review." and the
review it just built opens, while the row behind it becomes "◌ Changes
available · Next: Review changes" so **Review** reaches the same review again
without re-checking. A check that finds nothing says "Nothing to review." and
leaves you on the list. **Review** also appears on its own
when a root's changes need attention;
legacy candidates use **Review migration**. **Pause** and **Resume** control an
active root; a paused root reads "Ⅱ Paused · Next: Resume". **Resume**
re-activates the root and runs the same check as
**Check changes**: a root with nothing changed returns to "✓ Up to date as of
HH:MM · Next: Check changes", and edits made while it was paused surface as "◌ Changes
available" or "⚠ Needs attention · Next: Review changes" (task-32519). A check
the app cannot run says so on the row itself: the row flips to "⚠ Needs
attention · Check failed — <reason> · Next: <action>" and offers that action's
control — Check changes on a paused root reads "Check failed — folder is
paused · Next: Resume" and puts **Resume** first. The reason is never the
exception text; where the cause cannot be named, the row states its category.
A failure changes only what the row SAYS, never what it is: a disconnected or
externally-held folder keeps its Check disabled with its own reason, and never
grows a Pause it should not have. The next action you run on that root clears
the failure and the line goes back to restating the row (task-32534).
**Retarget** and **Disconnect** carry their reason in the control's own label
— "○ Retarget unavailable — not in this release", "○ Disconnect unavailable —
not in this release" — and one line under the whole root list states it once
for both, with what it costs you: "Retarget/Disconnect unavailable — not in
this release; nothing on disk or in Notes changes." (task-32545).

A paused folder keeps its receipts: reading them is a read, so pausing one
folder never blanks the list for the others (task-32534).

**Every root here is titled with its folder's name** — the display name you
typed when you set it up, which is also the name the Notes list shows for the
root's managed folder ("Vault sync" in both places) — so two roots are told
apart by their titles alone. The row still carries no path. A migrated legacy
candidate has no typed name until you activate it: its row reads "Migrated
notes — review to finish setup", and once activated it takes its new folder's
"Migrated notes …" name. A root still being set up — no managed folder yet —
reads "Sync folder (setting up)". A root that has a folder but no name to show
says which: "Sync folder (missing from Notes)" when its folder is gone from
Notes (or its name is blank), "Sync folder (name unavailable)" when the app
could not read the name; the status line beside it says what to do. Names are
read when the app starts and when a folder is set up: rename a sync-managed
folder in the Notes tree and its row keeps the old title until the next start.

**Receipts**, under the root list, is where the writes lasting sync performs
on its own show up: the newest 20 across every listed root, newest first, as
"when · what happened · file · note" — "2026-09-14 07:34 · Wrote note to file ·
People/Sam.md · Sam" for a note you edited in Chatbook, "Updated note from
file" for a file you edited on disk (task-32534).

Saving a note in Chatbook is enough to produce that "Wrote note to file" row:
a note inside an active root that you save in the editor is written to its
file on its own, on the same terms a file you edit on disk is read into its
note — you do not have to run **Check changes**, and the row reads
"✓ Up to date as of HH:MM", the minute that write was confirmed, because by
then it is (task-32604). **Ctrl+Q** from the Library right after such a save waits for that
write before Chatbook exits — up to five seconds; if the folder's sync takes
longer than that, the file receives the edit when Chatbook next starts. This
depends on lasting
sync still running: if it has stopped, nothing is carrying changes either way
and the root's row says so — "⚠ Sync stopped · Next: Check changes" — instead
of claiming to be up to date. (That one label still appears only on the next
redraw — on opening **Manage sync folders**, or on returning to the Library:
a runtime that has stopped publishes nothing, so a list you are already
sitting on cannot learn it from a publication.) The same row also appears
for one folder on its own when lasting sync is running but nothing is
watching that folder — the fact is read each time the list is drawn, never
remembered from the last status — and **Check changes** on that row starts
watching it again; when watching resumes on its own (a backup finishing,
say) the row returns to "✓ Up to date as of HH:MM" without a keypress
(TASK-34000.50). Until this the note side
produced no signal at all: Chatbook watches the folder, not the notes
database, so a note you saved stayed in Notes only and its file kept its old
bytes until something else touched the disk — with the row reading
"✓ Up to date" the whole time.

A folder that is **held for attention** — a sync entry that could not be
completed, a conflict or deletion waiting for your review, a failed pass — is
not syncing in either direction until you act, and every Notes surface says so
rather than only Manage sync folders (TASK-34000.2, review finding N-02): the
folder's tree row reads "⚠ Needs attention" in place of "⇄ Sync managed" (and
"⚠ Sync stopped" for a folder that is healthy but not being watched -- the same
row Manage sync folders shows for it, TASK-34000.50), the
list's idle status reads "Library notes · ⚠ A sync folder needs attention ·
Next: Open Manage sync folders." in place of "Ready", and the editor of any
note in that folder says "Saved HH:MM in Notes · ⚠ Sync needs attention" with
"In a synced folder · ⚠ Sync needs attention · …" under its title. These
surfaces follow the folder as its state changes — a hold that a background pass
produces while you sit idle (a disk edit colliding with a note edit, say)
reaches them within a moment, without a keypress — and they return to the
healthy wording the same way once the folder is resolved. Typing in a synced
note never holds its folder: Chatbook saves the note while you type, and a save
that lands while the folder is still writing the previous save to the file is
settled on the next pass with no click -- the folder records the baseline it
can prove (the journal's digest and the bytes on disk) and writes the newer
text; while a long burst of typing is being caught up the row may briefly read
"◌ Changes available" before it returns to "✓ Up to date", with nothing to
press; a save never waits for a sync, on this folder
or any other, so a note switch, Back or a rail switch is never delayed by a
folder's pass (TASK-34000.51). The one thing that still waits is a change on
both sides -- the file edited on disk while the note was edited in Chatbook --
which is an ordinary "Both file and note changed" review, never a silent
winner. A file's line-ending convention is kept across an edit that leaves
no line ending at all -- a Windows-style CRLF file whose note shrinks to one
line stays a CRLF file, the next multi-line edit is written CRLF, and a
folder that an older build held on exactly this ("Recovery failed" after
such an edit) is healed by **Recovery** (TASK-34000.48). An ordinary edit —
Ctrl+End, then a word without Enter, in a vault whose files end with a
newline — completes: the written file is compared with what the note writes,
newline convention included. A folder still holding an entry an older build
left open reads "⚠ Needs attention · Next: Resolve recovery" in **Manage
sync folders**, with **Recovery** first.
**Recovery** closes the open entry at the baseline it can prove on disk and in
Notes, mutating neither side, then checks the folder again: whatever changed
on one side since is synced, and a change on both sides becomes an ordinary
"Both file and note changed" review (Keep both / Keep file / Keep note / Skip
for now) — never a silent winner, and never "Recovery failed — RuntimeError"
next to a **Check changes** that refused the same entry. A folder already
stuck that way by an earlier build is healed by the same button.

**The healthy row is dated.** Because not every way of writing a note tells
lasting sync about it yet (the list below), a root's healthy status names the
minute it was last confirmed — "✓ Up to date as of 14:32 · Next: Check
changes", in your local time; a confirmation from another day carries its
date, "as of 2026-10-03 08:25" — rather than standing as an unqualified
promise. The time comes from the runtime's own publication, so it moves when
a pass finishes, and the rows in **Manage sync folders** follow the runtime
while you are looking at them: a pass that finishes while you sit on the list
repaints its row within a moment, without a keypress (TASK-32633, review
finding N-03).

**Deleting a synced note holds its folder until you restore the note.**
Lasting sync never chooses a winner, and that includes a deletion: when you
delete a note that lives in a sync folder, the file stays on disk, byte for
byte, and the folder is held — the tree row reads "⚠ Needs attention", the
list "Library notes · ⚠ A sync folder needs attention · Next: Open Manage sync
folders.", and the root's row "⚠ Needs attention · Next: Review changes". The
delete prompt says so before you confirm: "Delete this note? Its file stays
on disk; the synced folder waits for review until the note is restored. Undo
will be available in the Notes list." Know what "waits" means in this
release: **Review changes** shows the deletion but cannot resolve it — every
choice on that row is disabled ("Deletion review is unavailable in this
release", TASK-34000.15) — and **nothing in that folder syncs in either
direction** while it is held. The one way back is to restore the note:
**Undo** on the receipt, or **Restore** in **Recently deleted**, in this
session or a later one. The folder then returns to "✓ Up to date as of HH:MM"
on its own, with no **Check changes**, and the file is unchanged throughout.
(Two narrow exceptions, with the same way out. If a backup is capturing the
profile at the moment you restore, the folder waits for your next change to
one of its notes, or for **Check changes**. And **Ctrl+Q** right after a
restore waits up to five seconds for the folder to be checked; on a folder
that takes longer, the next session opens with it still held.)
A folder that opens held — Chatbook was quit while it needed attention, so
the next session starts with the row already reading "⚠ Needs attention" —
is also healed by **Review** on that row once the cause is gone (the note
restored, or the file back on disk), or by **Recovery** on an open entry,
and either leaves the folder watched: the next edit on disk reaches the note
and the next save in Chatbook reaches the file, with no **Check changes**
(TASK-34000.50; before it, that row turned to "✓ Up to date as of HH:MM"
while nothing was watching the folder, and only Check changes started it).
The folder keeps syncing both ways after that: the first edit you make to
that file *on disk* flows into the restored note on the next automatic pass,
exactly as it did before the delete. (The same holds after a keywords-only
save or a rename of the note in Chatbook — none of those count as a change to
the note's text, and none of them stop the next file edit from landing.) If
you edited the note's text in Chatbook *and* the file on disk, the folder
stops for the ordinary "Both file and note changed" review instead; nothing
is chosen for you.

**What this covers, exactly: editing an existing synced note in the Library
note editor, and deleting or restoring one through the Library.** Those are
the writes into a note that tell lasting sync anything. These do not, and
their files stay as they are until you run **Check changes**, until something
changes on disk, or until the next start — while the row goes on reading
"✓ Up to date as of <the last time it was confirmed>":

- **New note**, including a new note created straight into a synced folder.
- **Save as Note** in Console, and the same action on text you select inside a
  Console message.
- A note written by Research, by an ingest job, by an MCP tool, by the
  assistant's own `create_note` or `update_note` tool, by the Library
  `library_save_note` tool, by **Import once** over a note that already
  exists, or by a chatbook import.
- Deleting a note from anywhere other than the Library (a Research quick
  note's delete, for instance).

None of those is new — none of them ever told lasting sync anything, and
task-32604 changed only the editor — but none is fixed either, so if you need
one of them on disk now, run
**Check changes** on the root and apply the review. Chatbook also watches only
while it is running: a root whose notes changed outside a session reconciles
at the next startup check.

Tab moves through the root controls; the footer names the one you are on
("enter check changes", "enter pause") instead of a generic "run action", and
disabled controls keep the generic chip because Enter does nothing there
(task-32545). Tab stays inside the Keep a folder synced pane, as it does in
Import once: from the review it reaches **Activate reviewed root**, where
Enter activates the root, and **F6** and **Esc** are the ways out.

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
your home directory the first time. From these three doors it opens with
that **File name** field ("File name or path") already focused, so the
first thing you type goes into it rather than into the file listing: press
**Enter** to browse into the typed path, or **Select folder** to use the
folder being browsed. (This does not describe the Folder files door below,
whose picker starts on its own pre-filled **Folder path** field instead.) Its three actions
— **Open**, **Select folder**, **Cancel** — show heavy bars down their left
and right edges while focused, so which one Enter would press is visible
without relying on colour (task-32540). Its path trail across the top shows
the first folder, then `…`, then the last few, on one line; each crumb's
tooltip is the absolute path it goes to, and the full location is spelled out
above it (task-32554). The field
arrives empty and stays empty as you browse — the "Folder path" field that
arrives pre-filled with the directory being browsed, and selects that value on
the click that focuses it, is the Folder files root picker's
([File notes](file-notes.md)); here `Ctrl+A` selects whatever you typed. An
invalid path shows its reason on a row under the field ("The file must exist")
and leaves the dialog open. **Select folder** picks the folder row you last
clicked if you clicked one, otherwise the folder being browsed.
Once a folder is picked, the confirmation line shows its full path
(elided in the middle only when the path is longer than the pane is wide,
keeping the folder name itself visible), not just its name — and it is the
only line that states the selection: the status line above it says what to
do next ("Check the selection to see what will be imported.") rather than
repeating the count (task-32554).

If that folder is an Obsidian vault, a second line under the confirmation says
so before you check anything: "Obsidian vault", how many notes the scan would
read, and what it skips — the vault's own folders when they are there
(`.obsidian/`, `.trash/`, `Templates/`) and empty files, which are always
skipped. When some of those files have already been imported it says how many,
and when a lasting-sync root already covers the folder it says "this folder is
already kept in sync" — both of them here, where you can still pick a
different folder. A folder that is not a vault gets no extra line at all
(task-32641).

Picking the folder puts focus back inside Import once, on the confirmation
itself. From there **Tab** walks **Change selection** → **Clear** → **Check
selection** → **‹ Notes** and round again: Tab stays inside the Import once
pane rather than leaving for the Library rail, exactly as it stays inside the
note editor. **F6** and **Esc** are the ways out (task-32540).

Choose **Check selection** to build a read-only review. The review takes the
pane while it is open — the Notes list steps aside and comes back when you
leave. The status line above it states the total ("Review 66 sources before
import.") before you approve anything.

Each source is one line — path · what will happen · where it lands — with its
**Skip** and **Create new** controls beside the path. The middle clause names
the outcome directly ("Create 1 new note: Monday · keywords daily, focus · 2
links") rather than the column it came from (task-32554). The path gives way
first if the line is too long (elided in the middle), because the half that
decides anything is the outcome: the resulting title, its keywords and its
link count.

Rows are grouped by outcome (**New**, **Unchanged repeat**, **Changed
repeat**, **Uncertain match**, **Unsupported**, **Skipped**, **Empty**,
**Failed**) and each group header carries **Skip all on this page** and, where
the group can create notes, **Create all on this page** — both act on exactly
the rows that page's heading counts, so a later page keeps its own choices.
**Unchanged repeat** has nothing to create, so its header carries **Skip all
on this page** alone.

Importing the same folder twice recognises every source you already
imported, whichever kind it is: a `.md` file, and equally a `notes.csv`, a
`meta.yaml` or a `scratch.txt` — a source that parses into several notes is
matched on all of its records together. Unchanged ones land under **Unchanged
repeat** with **Skip** already chosen; edit one and it moves to **Changed
repeat**. A source that still matches but now yields a different number of
notes than last time cannot be lined up record for record, so it lands under
**Uncertain match** for you to decide. Multi-record sources offer **Skip** and
**Create new** only: there is no one note for **Update existing** to write to.
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
content** to see it. A row that says "No change." never carries a
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
current item; completed items are not rolled back. Escape leaves the import
running (use **Cancel import** to stop it): it returns to the Notes list, as
**‹ Notes** does, and the footer reads `esc back to notes`, as it does
before the import starts. While the
selection is still being checked, Escape cancels the check instead (`esc
cancel`) and keeps you on the import with your selection. Progress counts *planned
changes* — one per note a source creates, one per source otherwise — so a
two-note CSV is two of them; the line says so ("67 of 67 planned changes
complete").

The receipt states what happened once, in plain words — "59 notes created ·
8 files skipped · 54 links to imported notes rewritten" — with the two counts
reconciled in the line beneath it ("67 planned changes from 66 reviewed
sources."), and a **Skipped (N)** disclosure lists each skipped path with its
reason. The links figure counts every `[[link]]` that found a note the same
batch created, across all of them, which is why it can be larger than the
number of notes that carry one. When the batch imported notes containing
Obsidian embeds, the receipt says so too ("3 embedded files left as text").
The receipt's own action then takes you to the Notes list holding what it
created, with the count on its label, and leaves focus where leaving Import
once does: on the first note row. A file the app skipped for you — an unchanged repeat, an empty or
unsupported source — keeps its own reason there; only a row you set to Skip
yourself reads "Skipped by you." A partial receipt states what finished. Retryable failures show
**Retry N failures**; a cancelled batch with unfinished items shows **Retry
unfinished items**. **‹ Notes** or Escape may hide a running import without
stopping it; the list then offers **View import** or **Continue import** until
it settles, and **View import** reopens the running import with its progress
and **Cancel import**. Focus lands on the progress, not on **Cancel import**,
so pressing Enter there does not stop the import; **Cancel import** is one Tab
away. **Last import** reopens the same-session receipt afterward.

#### Obsidian vaults

If the folder you chose holds an `.obsidian/` directory, the selection line
says so as soon as you pick it (see "Import once" above), and the review shows
an **Obsidian vault** toggle, on by default, and one line saying what it does.

Windows works the same way: the Windows discovery adapter detects the vault and
skips its own folders exactly as the POSIX one does.

With it on:

- `.obsidian/`, `.trash/` and `Templates/` are listed under **Skipped** as one
  row each, naming the vault reason ("Obsidian configuration — skipped…"),
  rather than one row per file inside them.
- YAML frontmatter is read: `title` becomes the note title, and `tags` (or
  `keywords`, the key this app's own older exports wrote) and
  `aliases` become keywords (there is no separate alias field, and keeping them
  as keywords is what makes the note findable by its alternate names). An alias
  is an alternate *name*, not a tag, so it is stored as `alias: <name>` — in
  Info you can tell the two apart, and searching for the name still finds the
  note. The frontmatter block is removed from the note body — unless it is the
  whole file, in which case the note keeps it and still takes its title and
  keywords from it. Any other property in that block (`mood`, `status`,
  `rating`) is not imported, and the review row says which ones ("… · not
  imported: status"), so nothing disappears silently — that includes a key
  YAML does not read as text (`Yes:` is reported as `True`), a `title` that is
  not text, and a `tags` list the import could not keep ("tags (more than
  100)"). A `title` that is a number or a date (`title: 2026`, `12:30`) is used
  exactly as written. Your own `date:` or `id:` property is reported like any
  other — a daily note's `date:` beside nothing but its `tags:` too; only a
  block Chatbook exported itself, one that names a `note_id` (**Export
  Markdown**) or an `id` with `created_at` and `updated_at` (a bundle), drops
  its `date` and ids silently, because they are the export's stamps, not
  yours.
- Outside a vault — a picked file, a plain folder, or the toggle off — the
  block is read and removed only when every key in it is one of the above
  (`title`, `tags`, `keywords`, `aliases`), or when the block is one Chatbook
  exported (it names a `note_id`, or an `id` with `created_at` and
  `updated_at`, so its `date` and ids are the export's own): that is how a
  note you exported with **Export Markdown** or in a bundle comes back as it
  left. Any other block — your own `Summary:` or `Decision:` properties, or a
  daily note's `date:` beside its `tags:` — stays in the note body exactly as
  written, and nothing is dropped.
- An embed — `![[diagram.png]]`, or `![[Another note]]` — is not a link and is
  never rewritten: Chatbook has nowhere to render the embedded file, so the
  note keeps the `![[…]]` exactly as you wrote it and Preview prints it as
  text. The review row says so where it applies ("… · 2 embeds show as
  `![[…]]` text, not the files"), and the receipt repeats the total, so the
  "Unsupported · Image — not a note" row and its consequence are on the same
  screen.
- `[[wikilinks]]` and `[[link|alias]]` whose target is imported in the same
  batch become note links; a link to anything else stays as plain text, and a
  `[[link]]` written inside a code block or backticks is left alone; a
  `` ``` `` or `~~~` fence that is never closed counts as code to the end of
  the note. A linked
  note keeps its wikilink and shows the linked note's title, with the
  identifier behind it — `[[Reading/Zettelkasten|Zettelkasten — overview]]`
  followed by `(note://…)`. Preview renders it as the title alone, an exported
  file is still a working Obsidian link, and importing that file again
  recovers the same link rather than stacking a second identifier on it.

Keep a folder synced offers the same **Obsidian vault** toggle, default-on,
once the folder you chose holds an `.obsidian/` directory — see "Import once
vs Keep a folder synced on the same folder" below for the one thing the two
paths do differently with it. There, turning the toggle **off** lasts only
until you quit Chatbook: the choice is not stored, so a vault is offered the
toggle again, on, on the next start, and a later check of that root skips the
three folders again. The toggle is offered while you are setting the folder up
and nowhere else — once the root is active, **Manage sync folders** has no
switch for it — so if you want those folders synced, import them with Import
once instead. Nothing already synced changes when it comes back on —
the pass only ever leaves files the sync has never taken alone.

Turn the toggle off to import the vault exactly as any other folder — every
directory walked, frontmatter left in the body (unless every property in the
block is one Chatbook reads — `title`, `tags`, `keywords`, `aliases` — or the
block is one Chatbook exported, naming its `note_id` or its `id`,
`created_at` and `updated_at`, in which case it is read and removed as with
the toggle on), links left as text. The config
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

#### The same import, keyboard only

No step needs the mouse (task-32540). From the notes list:

1. **Tab** to **Add from files…**, **Enter**.
2. The chooser opens with **Import once** focused — **Enter**.
3. The picker opens on its path field — **type the folder path** (the listing
   follows you as you type), then **Tab**, **Tab** to **Select folder** and
   **Enter**. (**Ctrl+S** does the same thing from anywhere in the dialog.)
4. Focus lands on the confirmation. **Tab**, **Tab**, **Tab** to **Check
   selection**, **Enter**.
5. The review opens with **Import selected items** already focused — **Enter**.

The receipt is then on screen. Tab stays inside Import once throughout —
**F6** moves between panes and **Esc** leaves.

### Set up lasting folder sync

1. Close any older Chatbook version using this profile, then restart the
   cutover release.
2. In the notes list, click **Add from files…** and choose **Keep a folder
   synced**.
3. Click **Choose folder…**; type a path into the **Folder path** field and
   either press Enter (browses into it) or click **Select folder** (uses the
   folder being browsed). Choose a direction and local
   destination.
   Server sync remains unavailable until its separate capability is
   installed.
4. Choose **Check changes** and review the exact safe, attention, skipped, and
   deletion-like effects. If the folder cannot be used, the status line under
   the pane's "Add files to Library notes" heading says which rule it broke
   and what to do; choose **Choose folder…** again and check the new one.
5. Choose **Activate reviewed root**. If the review is stale, choose **Check
   again** instead. **Manage sync folders** appears in the notes toolbar once
   a root is active. The receipt reads "N applied · listed under Receipts" and
   names where to see them (task-32545). The receipt's bar offers **Manage sync folders** and
   **‹ Notes**; a second folder is set up from **Add from files…** again.
   Escape from these pages returns to Notes, like **‹ Notes**; from a review
   you opened in **Manage sync folders** it returns to that list, like the
   page's **‹ Sync folders**.
   Choosing **‹ Notes** returns to a Notes list that already
   counts the synced notes and shows them under a **⇄ Sync managed** folder
   named after the display name — no restart needed (task-32518).

#### Import once vs Keep a folder synced on the same folder

Run over the same Obsidian vault, the two paths skip the same files for
the same reasons: `.obsidian/`, `.trash/` and `Templates/` while the
**Obsidian vault** toggle is on, and an empty or whitespace-only file
whatever the toggle says. (Keep a folder synced does not remember the toggle
across a restart — see "Obsidian vaults" above.) Both read the frontmatter, so `title` becomes the
note's title and `tags` (or `keywords`) and `aliases` become its keywords. A synced note bounds
both: a tag or alias longer than 256 characters is dropped (half an alternate
name is a name nothing has, and the note keeps its other keywords), while a
`title:` longer than 4,096 characters is cut to the first 4,096 (a title is
the note's only name, so cutting it keeps the note findable where dropping it
would silently rename the note to its file name). Neither costs you the note,
and neither stops the rest of the folder from syncing.

**Running both on the same folder does not import it twice.** If you have
already used **Import once** on a folder, keeping that same folder synced
leaves those files alone rather than making a second note for each one: the
review lists them as skipped, reading "Already imported by Import once — left
as it is", and the counts line says `0 safe`. Only the files the import never
turned into a note — added since, or skipped at the time — are created and
synced. A note you have since deleted stops counting, so the file is created
and synced on the next check.

The consequence worth knowing: **those files are not synced.** They are
ordinary Library notes with no connection to the folder, and Chatbook will
not carry your edits either way for them. To put an already-imported vault
under lasting sync, delete the notes Import once made from it first and then
run **Check changes** again — each file is then created once and kept synced.
**Know the cost before you start:** Chatbook has no bulk delete for notes, so
that is one note at a time, each with its own confirmation — select mode can
export, move and re-keyword notes in bulk, but not delete them. For a vault of any size it is worth
setting the sync up on a folder you have not imported, rather than undoing an
import of fifty notes by hand.

**The other order is not protected yet.** Running **Import once** on a folder
you are *already* keeping synced still copies every file into a second set of
notes — Import once recognises its own earlier imports, not sync's notes. Set
the sync up first and leave Import once alone for that folder.

Two things still differ:

- **The frontmatter block stays in a synced note's body**, byte for byte,
  where Import once removes it. Keeping a folder synced is two-way: editing
  the note in Chatbook writes the body back to the file on disk, so a
  stripped block would delete the properties out of your vault on your next
  edit. **Folder files** keeps the block for the same reason.
- **Synced notes do not keep the vault's folder tree.** Every synced note
  sits directly in the sync-managed folder, and the review says so — each
  row's destination is that folder. Import once reproduces the folder
  hierarchy. The file's own folder is still visible on every review row, in
  its path, and in the note's title where the frontmatter names one.

Existing legacy evidence appears as a paused candidate. Open **Manage sync
folders**, choose **Review migration**, inspect the current dry-run, and
activate explicitly. The migration never inherits a legacy conflict winner or
automatic-sync setting.

### Use a note in Console
1. Open the note and click **Use in Console**.
2. The note is linked to your active workspace on the way — a new note
   belongs to no workspace, and that is what the hand-off needs — so the
   status line reads "Use in Console complete — Linked to *workspace* ·
   staged in Console." The line says what actually happened: "Already
   linked to *workspace* · staged in Console." when the note was a member
   already, and "Staged in Console." when no linking was needed at all.
3. You land in the Console with the note staged as context and the
   prompt "Use this note as context and help me work with it." ready to
   send or rewrite. On a profile with no provider set up yet, the note is
   still staged and the Console shows its own setup card; add a provider
   from there and the staged note is waiting.

If the hand-off cannot be made, the editor's status line is the only
message, and it names what stopped it and what to do — for example
"Can't use this note in Console — the workspace registry is unavailable.
Next: restart Chatbook, then try again." Opening another note, or saving
this one, clears it. A hand-off that fails leaves no workspace link behind
— if it linked the note on the way, it removes that link again, and says
so if it could not.

### Capture a Console answer as a note
The return leg of **Use in Console**. In the Console, select an assistant
reply, click **More…**, then **Capture as note**. The reply is saved here
immediately — titled with its first line of text (a leading code fence or
heading mark is dropped), holding the answer verbatim — and
a "Saved to Notes" receipt offers **Open note**, which lands you on that note
in the editor.

A captured note carries three keywords: `console`, `conversation:<id>` and
`message:<id>`. They are ordinary keywords, so filtering the list on
`conversation:<id>` finds every answer kept from one chat. Capturing is blocked
while the Console chat is temporary — a temporary chat promises nothing is
written locally, and a note is a local write.

### Export a note as Markdown
1. Open the note, choose **Info**, and click **Export Markdown**.
2. Choose a destination in the "Export Note as Markdown" dialog — the
   toast confirms "Note exported successfully to ~/exp/\<name\>" with the
   folder and the file.
3. If that file already exists, the "Replace existing file?" prompt asks
   first. **Cancel** is focused, so Enter or Escape keeps your file as it
   is; choose **Replace** to overwrite it.

The front matter is real YAML (`title`, `date`, `tags`, `note_id`): a title
with a colon or a leading `*` is quoted, and `tags` is a list, so Obsidian,
Pandoc and **Add from files…** all read it back.

### Undo a deleted note

1. Confirm **Delete** in the note editor.
2. In the Notes list, find the "✓ deleted · \<title\>" receipt.
3. Click **Undo** to restore the note, or **Dismiss** to leave it deleted.

Keyboard focus lands on **Undo** as soon as the delete is confirmed, and the
footer reads "enter undo delete" while it is there, so recovering a note you
did not mean to delete is a single **Enter** — no hunting for the receipt.
Tab from there reaches **Dismiss**, which the footer names as well.

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

The list pages 20 at a time with the exact total — for example
"1–20 of 37 · Page 1 of 2" — and **Previous** / **Next** under it move
between pages; while every deleted note fits one page, only the range is
shown. Restoring a row keeps you on the same page: the rows below it move up,
and focus lands on the row now in its place — or on the new last row when you
restored the bottom one. Restoring the only row left on the last page shows
the page before it, with focus on its first row. If a page cannot be read, the
view keeps what it shows, says "Couldn't load recently deleted notes." and
offers **Retry**.

**Delete forever** (x) removes a note for good after a confirm that names it.
It shows on the focused row only, so every other row keeps its whole title;
with the mouse, click a row's title to focus that row (this restores nothing),
then click its **Delete forever**.
A note kept in sync with a folder is refused with "This note is kept in sync
with a folder, so it can't be deleted forever here."; this release has no way
to take one note out of its folder's sync. The confirm opens on **Cancel**,
and **Escape** cancels it without changing anything. There is no Undo
afterwards: the note, its keyword and folder links and its earlier versions
are gone, and focus lands on the row now in its place.

## Keyboard & commands

| Key | Action |
|---|---|
| **n** / **Ctrl+N** | Makes a new blank note and opens it — no chooser in between. Both work on the Library landing (no row selected yet) as well as inside the Notes workflow. The notes list advertises the bare **n**; **Ctrl+N** additionally works from inside the filter box, where a printable key would type instead. |
| **/** | Focus the note filter ("find note"), without typing a literal "/" into it. Once the filter has focus, "/" is an ordinary typeable character rather than an accelerator — a second "/" adds a literal slash, since a filter can legitimately target a folder-style path such as "Work/Q3". The footer drops the "/ find note" chip while a text field has focus, for that reason. |
| **g** | Go to the folder tree beside the list — focus lands on the selected placement when there is one, otherwise on the first folder row. |
| **e** (in select mode) | Export the checked notes. Advertised only once at least one row is checked, which is also when the **Export selected** button stops being disabled. |
| **Space** (in select mode) | Toggle the focused note — check it or uncheck it, as Enter on its row does. Only on a note row: on Done, Select all, Clear, Export selected or Actions… it does nothing, a pane grip keeps its own Space (it collapses or expands its pane, as outside select mode), and in the filter box it types a space. |
| **↑ / ↓** (Notes tree) | Move through folder rows, note rows and the "more" pager in the order shown — one list, top to bottom. Enter on a "more" pager loads the next page and puts focus on the first row it added. |
| **→ / ←** (on a folder row) | Expand the folder; collapse it, or jump to its parent folder when it is already collapsed. Selecting a folder (Enter) stays separate from these. |
| **Enter** (on **Sort**) | Opens the chooser with the current choice focused. **← / →** (or ↹ Tab) move between the choices without choosing, and Enter picks the focused one; Enter on a choice or Escape returns focus to **Sort**. |
| **Escape** | Focus the rail (in **Recently deleted**, go back to the list). In the filter: leave the field, keeping what you typed; the applied filter is unchanged. |
| **r** (in **Recently deleted**) | Restore the focused row |
| **x** (in **Recently deleted**) | Delete the focused row forever, after a confirm that names it |
| Enter (in "Filter notes… (Enter)") | Apply the filter; Enter again on the same filter opens the first result |
| ↓ (in the filter) | Focus the first result row. Text you have not applied yet is applied first, the way Enter applies it, and focus lands on the first row of the new result; with no result, focus stays in the filter. An emptied box clears the filter, as Enter does, and focus lands on the first folder or note of the list — the same row ↓ reaches from an empty box with no filter applied. Any key or click while the result is still loading — including F6, Shift+F6 and Tab — or a dialog opening over Library cancels that move to the first row. A mouse-wheel scroll that moves a pane does not: the pane takes the wheel, and focus still lands on the first row when the result paints. |
| **‹ \<note\>** (note heading) | After following a link or a backlink, returns to the note you came from (one hop) |
| ↑ / ↓ (New note view) | Move between **Blank note**, **From a template…** and the template rows it opens |
| Enter (New note view) | Create from the focused row |

The footer advertises these as `n new note | / find note | g go to folder |
esc focus rail`. While a text field has focus the printable ones are literal
characters, not accelerators, so the footer says so the way the rest of
Library does. In the filter it reads `typing in field | esc focus rail` and
names no keys "after esc", because Escape there moves the caret into the
rail's "Search Library…" box, another text field (every Library list filter
behaves the same). In that search box Escape leaves for the list, so the
footer adds `after esc: n new note · / find note · g go to folder`. A note's
Title, Keywords and body announce themselves the same way — `typing in field
| esc back to notes`, plus `ctrl+end end of note` in the body. Below 64
columns the Escape chip comes first, because the footer may have room for
only one chip there. In select mode the tier becomes
`space toggle | enter select note | e export selected | esc done`, with the
**space** chip present only while a note row has focus and the **e** chip only
while something is checked; with the Actions… strip open it reads
`enter choose action | esc cancel`. **Ctrl+S** belongs to the note's pane, not
the list: the list's footer never shows it, and pressing it there does nothing
(see Editor keys). In Folder files it writes the file now. Global navigation
keys live in the [guide index](../index.md).

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
- **"Use in Console" delivers the note's real content, not just its
  title** — your next send in Console carries an excerpt of the note
  body itself.
