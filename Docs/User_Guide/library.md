# Library — Your content hub: sources, search, and imports.

## What this screen is for

Library is where everything the app knows about lives: media you've
imported, conversations from Console, notes, prompts, skills, and
Collections — plus search and RAG over all of it, and the import/export
tools that move content in and out. Reach for it to add source material,
find something you saved, or hand a bundle of sources off to Console or
Study. This page is the orientation tour; the details live on eight child
pages:

- [Media & conversations](library/media-and-conversations.md) — browse imported media (with the media viewer) and your Console conversations.
- [Notes](library/notes.md) — the notes list, editor, templates, and the Notes sync panel.
- [File Notes](library/file-notes.md) — the folder-backed File Notes workspace and its Session Git panel.
- [Prompts](library/prompts.md) — saved prompts: list, editor, import, and Console insert.
- [Skills](library/skills.md) — skill packs: import, editing, and the trust/approval flow.
- [Collections](library/collections.md) — local Collection records for saved content.
- [Search & RAG](library/search-and-rag.md) — the Library Search/RAG canvas, evidence, and the Console handoff.
- [Import & export](library/import-and-export.md) — the Import media flow and the Export bundle (.zip) canvas.

## Getting there

- Press **Ctrl+3** from anywhere, or click **⌃3 Library** in the nav bar.
- **Ctrl+P** → "Tab Navigation: Switch to Library" in the command palette.
- Old destination names still find it: the six retired screens —
  **notes**, **prompts**, **skills**, **ingest**, **research**, and
  **media** — now live inside Library, and typing any of them into the
  palette routes here. (Typing **search** or **study** also surfaces the
  Library command — those words are aliases for it, and picking the hit
  lands on Library. The command palette's "Media & Content: Open Media
  Library" and "Quick Actions: Search All Content" entries are deep links
  into Library's Media and Search/RAG rows, not separate screens; Study has
  no palette command at all — reach it from Library's hand-off buttons.)
  The palette also offers "Tab Navigation: Library — Skills", which lands
  directly on the Skills row.

## Get started on a new profile

A new profile starts with a compact rail: **Import…**, **New note**, and
**Explore all tools**. While Library checks its sources, it says
**Checking existing Library content…** instead of claiming that the Library is
empty. If a source is unavailable, the same actions stay enabled and one
**Retry source check** action appears.

```text
new profile
    |
    v
checking sources -- usable content found --> full Library (permanent)
    |
    +-- all sources authoritatively empty --> Get started
                                                |  Import...
                                                |  New note
                                                +  Explore all tools
                                                       |
                                                       v
                                                full Library (remembered)
```

**Explore all tools** reveals the complete rail immediately and remembers that
choice independently of which rail sections are open. While the expanded
Library is still authoritatively empty, **Back to Get started** is available.
Adding any usable content permanently graduates the profile to the full
Library; deleting that content later does not hide tools again.

Compact presentation never blocks navigation. Deep links and command-palette
routes, including **Tab Navigation: Library — Skills**, can open a tool that is
not shown in the Get started rail. Existing profiles without this preference
open the full Library.

## Returning to a populated Library

At 120 columns and wider, the landing helps you resume without pretending it
knows more than the source owners do:

```text
+----------------------+-----------------------------------------------+
| full Library rail    | Continue                                     |
|                      |   [ Return to the last applied source scope ] |
| Browse / Create /    | Needs attention                              |
| Study / Import       |   current recoverable problem       [Retry]  |
|                      | From your Library                            |
|                      |   Database note / Media / Conversation       |
|                      | Quick actions                                |
|                      |   [Import…] [New note] [Search]              |
+----------------------+-----------------------------------------------+
```

- **Continue** remembers the last eligible source list separately from the
  currently open route. It appears only after that source's full scope was
  authoritatively applied. Returning from an item or detail view resumes the
  list scope, not a guessed item selection; the source read reports any page
  clamp or deleted content.
- **Needs attention** shows at most one current, recoverable problem from this
  Library screen and reuses its existing **Review** or **Retry** action. It is
  session state, not a promise that the warning survives restart.
- **From your Library** uses cached summaries in the fixed order **Database
  Notes → Media → Conversations**. Missing or unresolved sources are omitted;
  the order does not imply that items were ranked against each other.
- **Quick actions** are **Import…**, **New note**, then **Search**. They use the
  same guarded destinations as the rail.

At compact widths the landing canvas is hidden and the rail remains the
navigation owner. If focus was on Continue, recovery, a cached summary, or a
quick action when the terminal became compact, focus moves to that action's
matching rail destination. Widening restores the landing control only if you
did not choose a newer rail target in the meantime.

## Layout tour

```text
+----------------------+------------------------------------------+
| Library rail         | Canvas                                   |
| navigation and tools | landing, list, viewer, editor, or import |
+----------------------+------------------------------------------+
| context-sensitive keyboard hints and status                    |
+-----------------------------------------------------------------+
```

- **Header line** — reads **Library | Local**, or **Library | Server:
  \<label\>** when a server runtime is configured.
- **Left rail**, top to bottom. A new empty profile first sees the compact
  Get started controls above; the complete rail below appears after Explore or
  graduation:
  - a **Navigation** heading with **Collapse** at the opposite edge. Collapse
    hides the rail without changing the selected destination, search query,
    section disclosures, or canvas. The slim **Nav** handle expands it again;
    it is keyboard-focusable and remains part of the **F6** pane cycle;
  - the **Import…** button ("Add files, links, and transcripts to
    your Library.");
  - the **Search Library…** box — submitting it lands on the
    Search / RAG canvas and runs your query (an empty submit just opens
    the canvas). Press **/** anywhere outside a text field to jump
    straight into this box; pressing **/** again inside it — or clicking
    into it when it still holds a query from before — selects the whole
    query, so the very next keystroke replaces it instead of landing
    wherever you clicked;
  - four sections — **Browse** (Media, Conversations, Notes, Prompts,
    Skills, Collections, Search / RAG), **Create** (New note, New prompt,
    New skill), **Study** (Study decks, Flashcards, Quizzes), and
    **Import / Export** (Import…, Export). Each row is one line: the
    title with its count, plus a dim plain-language gloss on the jargon
    rows (e.g. "Search / RAG — find all"), shown consistently across
    visits — a row's gloss never flickers on or off just because its
    count arrived. On narrow terminals the gloss drops first; a handful
    of rows (Conversations, Flashcards, Collections) then fall back to a
    short label ("Chats", "Cards", "Sets") instead of an ellipsis, so no
    row label ever cuts off mid-word and the count always stays visible.
    The three Study rows are hand-offs (they are a
    two-step trip out of Library), so they group under their own section
    and add a second "see what carries over" line — that click opens a
    Library-local staging canvas showing what will carry into Study, not
    the Study screen itself; **Continue in Study** inside that canvas is
    the click that actually leaves, and **Escape** returns to the hub. The
    selected row is marked **▸**, and the Flashcards row shows "due: N"
    instead of a plain count;
  - a **Details** section, collapsed by default (see below). Section
    headers toggle open (**▾**) and closed (**▸**).
- **Canvas** (the right pane) — there are no tabs here: the canvas swaps
  to match whichever rail row is selected. Before you pick one, a populated
  profile sees the returning landing described above. Import… and New note
  remain reachable with **i** and **n**.
- **Footer** — shows the keys that work where you are. The full rail offers
  "/ focus search"; Get started keeps focus on its visible actions. The
  landing adds "i import content"
  and "n new note" (single-letter accelerators for the hub actions);
  the Search / RAG canvas adds "u use Library
  context in Console", "enter select evidence", and "o open evidence";
  a Media/Notes/Prompts/Skills/Collections list adds "esc focus rail";
  that list's item viewer/editor (or the media viewer) adds "esc back to
  list" instead; the Export canvas adds "esc back to Media" (or whichever
  canvas opened it — "esc back to hub" from the rail); and a Study
  staging canvas adds "esc back to hub". Every hint is a per-key
  "key action" pair — the Notes editor, for example, shows "ctrl+s save
  note | esc back to notes".

One special case: selecting **Notes** adds a
**Library notes | Folder files** strip above the workbench. **Folder files**
swaps the canvas pane for the File Notes workspace. At 120 columns and wider
the rail stays beside it; on compact terminals the canvas becomes the single
visible stage so its controls remain on-screen. Escape (or the
**Library notes** link) returns to the notes list — see
[File Notes](library/file-notes.md).

## Features & controls

### Left rail

| Control | What it does |
|---|---|
| **Collapse** | Hides the wide navigation rail in place and gives the canvas the reclaimed width. The choice lasts for the current Library screen session. |
| **Nav** | Expands a manually collapsed rail and returns focus to **Search Library…**. On compact terminals, Library's existing one-pane routing takes precedence and the manual collapse returns when the terminal is wide again. |
| **Import…** | Opens the Import media canvas — see [Import & export](library/import-and-export.md). |
| **New note** | Opens the production note-creation canvas. It is shown directly in the Get started rail. |
| **Explore all tools** | Reveals and remembers the complete Library without changing section disclosures. |
| **Back to Get started** | Returns an explicitly expanded, still-empty Library to the Get started landing and compact rail, with focus on **Import…**. It is never offered after graduation. |
| **Search Library…** | Type a query and press Enter: lands on the Search / RAG canvas and runs it (empty submit just opens the canvas) — see [Search & RAG](library/search-and-rag.md). |
| **▾** / **▸** (section headers) | Open or collapse that rail section. |

### Browse rows

Media, Conversations, and Prompts replace an empty page's disabled paging and
selection controls with a useful next step. The exact total remains visible in
the title, but there is no meaningless “page 1 of 1” or “nothing to select”
mechanic.

```text
source really has no items          active filter has no matches
--------------------------          ----------------------------
Media (0)                           Media (0)
No media in your Library yet.       No media of type 'video'.
Import something to see it here.
[ Import media ]                    [ Show all types ]

Conversations (0)                   Conversations (0)
No conversations yet. Chat in      No conversations match 'draft'.
Console and it appears here.
[ Start in Console ]                [ Clear filter ]

Prompts (0)                         Prompts (0)
No prompts yet. Create or import    No prompts match "draft".
a prompt to begin.
[ New prompt ] [ Import... ]        [ Clear filter ]
```

A filtered empty page keeps its submitted type, query, or collection visible
until you choose the reset action. Loading and failed refreshes do not use this
empty presentation: they keep their status, pager authority, and **Retry** so a
previously empty page cannot hide an in-progress or recoverable request.

| Row | Opens | Details on |
|---|---|---|
| **Media** | The media list and viewer. | [Media & conversations](library/media-and-conversations.md) |
| **Conversations** | Your Console conversations, with preview and "Open in Console". | [Media & conversations](library/media-and-conversations.md) |
| **Notes** | The notes list/editor, plus the Library notes \| Folder files source strip. | [Notes](library/notes.md) |
| **Prompts** | The prompts list and editor. | [Prompts](library/prompts.md) |
| **Skills** | The skills list, editor, and trust panel. | [Skills](library/skills.md) |
| **Collections** | Library Collections (local records). | [Collections](library/collections.md) |
| **Search / RAG** | The Library Search/RAG canvas. | [Search & RAG](library/search-and-rag.md) |

### Create rows

| Row | What it does |
|---|---|
| **New note** | Opens the note-creation canvas: **Blank note** or a pick from "From a template" — see [Notes](library/notes.md). |
| **New prompt** | Opens a fresh prompt editor — see [Prompts](library/prompts.md). |
| **New skill** | Opens a fresh skill editor — see [Skills](library/skills.md). |

### Study rows

| Row | What it does |
|---|---|
| **Study decks** / **Flashcards** / **Quizzes** | Hand-off canvases that open the Study screen — see the next section. |

### Import / Export rows

| Row | What it does |
|---|---|
| **Import…** | The full import flow: path or URL, pre-flight check, per-type options, queue — see [Import & export](library/import-and-export.md). |
| **Export** | The "Export bundle (.zip)" canvas: package local content into a portable file — see [Import & export](library/import-and-export.md). Disabled in server mode. |

While a Local import is running, its queue row shows the parser's current stage.
A percentage appears only when the parser has a real bounded total; stages without
one intentionally show text alone. **Saving to Library** means parsing has handed
the result to the writer. These stage updates are best-effort and transient, so an
import may skip intermediate updates and does not resume from an earlier percentage.

### Details

Collapsed by default; click anywhere on the **Details** header — the label
text or the **▾**/**▸** chip — to open it. Opening it recomputes the
"DB sizes" line from disk (sidecars included), so the numbers you see are
current as of that open, not a reading cached at some earlier repaint.

| Group | Contents |
|---|---|
| **Status** | A "Source · Local" (or "Source · Server: \<label\>") line, and a counts row: "Notes N · Media N · Conversations N". |
| **Workspace** | "Active · \<workspace name\>" and "Handoff · \<summary\>" lines. |
| **Actions** | The buttons below, plus the note "Server sync WIP · local only". |

| Action | What it does |
|---|---|
| **Create local workspace** | Opens the same "New Workspace" dialog Console and Settings use — a prefilled "Workspace N" name, optional folders to bind (validated as added, with a Browse… picker), and a "Switch to this workspace" checkbox (checked by default). Escape cancels with nothing created. Server sync and ACP handoff remain WIP. A bound folder containing a `.SKILLS/` project skills folder is annotated "— contains N project skill(s)" and, after Create, offers a chained import prompt — see [Project skills](library/skills.md#project-skills-skills). |
| **Import sources** | Shown only while you have no workspace-eligible sources: "Open Library Import/Export to add workspace-eligible sources." |
| **Use in Console** | Stages a snapshot of your local Library sources ("Local Library Sources") into Console and takes you there. When it can't run yet, its tooltip says why — "Stage Library source context after Library finishes loading." or "Stage Library source context after adding notes, media, or conversations." |

### Study, Flashcards & Quizzes hand-offs

**Study is its own screen**, but it has no nav label and no palette
command — typing "study" into the palette surfaces *Library*. The hand-off
buttons below (and **Continue in Study**) are the way in.
The three Create rows in Library don't host study content; each shows a
small hand-off canvas that snapshots your Library sources. That first
click never leaves Library — it opens the staging canvas below; **Continue
in Study** inside it is the click that actually opens Study. Their purpose
lines:

- **Study decks** — "Plan study decks from Library sources."
- **Flashcards** — "Generate or review cards from Library sources."
- **Quizzes** — "Generate or resume quizzes from Library sources."

Each canvas shows the same five elements: the purpose line, a "Carries
forward: …" line naming up to three source titles (then "and N more."),
the ownership note "Generation and review run in Study.", a readiness
line ("Source snapshot is ready.", or a prompt to import sources or
create notes first), and a **Continue in Study** button ("Open \<X\> with
the current Library source snapshot, or globally when none is
available.").

Once you're on the Study screen, its header reads "Library ▸ Study" with
an "Esc: back to Library" hint — the nav bar shows no highlighted tab
there (Study renders none of Library's chrome, so boxing "Library" would
be misleading), and pressing **Escape** returns you to the Study decks
staging canvas above. (Reached from Home's **Review flashcards** instead,
the same screen reads "Home ▸ Study" and Escape returns to Home —
task-4011.)

## Common tasks

1. **Find anything you've saved.** Type into the **Search Library…** box
   and press Enter — you land on the Search / RAG canvas with results
   grouped as "Evidence · top 15 per source" (the number follows Settings ▸
   RAG's Default results; 15 on the shipped default profile). Narrow with
   the **Sources** scope toggles ([Search & RAG](library/search-and-rag.md)).
2. **Add your first file.** Click **Import…**, enter a file
   path or URL (or **Browse…**), review the pre-flight summary and
   options, then press **Start import**. The item appears under
   **Media** — full walkthrough in [Import & export](library/import-and-export.md).
3. **Create a note.** Click **New note** in the Create section, pick
   **Blank note** or a template under "From a template", and start
   typing — notes autosave (the meta line ends in "saved"). **‹ Back to
   list** returns you to the notes list.
4. **Hand your Library snapshot to Console.** Open the **Details**
   section, then under **Actions** press **Use in Console** — Console
   opens with a "Local Library Sources" snapshot staged as context.
5. **Open Study with your sources.** Select **Study decks**,
   **Flashcards**, or **Quizzes** in the Create section, check the
   "Carries forward:" line, and press **Continue in Study**.

## Keyboard & commands

Screen-level keys only — global keys live in the [guide index](index.md).

| Key | Action |
|---|---|
| / | Focus the **Search Library…** box when the full rail is visible (unless a text field already has focus). Get started has no hidden search target; use **Explore all tools** or a direct route. |
| u | Use Library context in Console — only while the Search / RAG row is selected (the footer hint appears only there) |
| ↑ / ↓ | Inside a Media, Notes, Prompts, or Skills list, move to the previous/next row (stops at the first/last row — it does not wrap) |
| Enter | Open the focused list row (same as clicking it) |
| Esc | Context-dependent — see below |

Entering a Media, Notes, Prompts, or Skills list (from the rail, or
returning from its item) focuses the list's first row, so ↑/↓/Enter work
immediately without tabbing to find it. Escape then reads the surface
you're on:

- **On the plain list** — Escape moves focus to the rail's **Search
  Library…** box in the full Library, or **Import…** in Get started; it never
  leaves the canvas or changes what's shown.
- **A pending bulk-delete confirmation on the Media list** (Select mode's
  "Delete selected", which swaps the list's toolbar for "Delete N
  selected items? This moves them to trash.") — Escape cancels it in
  place, exactly like its own **Cancel** button, instead of moving focus
  to the rail; the footer's hint reads "cancel delete" while it's armed.
  Confirming with **Delete** when only some items can be removed leaves
  the failed one(s) checked and focuses the first of them, rather than
  leaving nothing focused or landing on an item you never selected.
- **In an item's viewer or editor** (the media viewer; the Notes,
  Prompts, or Skills editor) — Escape returns to that list, re-focusing
  its first row, exactly like pressing **‹ Back to list**. A dirty note
  or prompt edit vetoes the exit the same way Back does.
- **Editing, deleting, or re-analyzing inside the media viewer** — the
  media viewer's Edit / Delete / Edit analysis forms have no dirty-edit
  guard, so a first Escape only discards that one form and returns to the
  plain read-only viewer (matching that form's own **Cancel** button); a
  *second* Escape from there returns to the list. The footer's hint
  changes to "back a step" while one of these forms is open, so it never
  claims "back to list" a step early.

Escape and Ctrl+S are also bound inside the skill editor specifically
(back to list / save) — see [Skills](library/skills.md). Escape also
returns Notes ▸ Folder files mode to the Library notes view, and is live
inside the File Notes surface's own panels and dialogs — see
[File Notes](library/file-notes.md). On the Study screen (reached via
**Continue in Study**), Escape returns to the Study decks staging canvas
here in Library.

## Related settings & docs

- `config.toml`: `[library]` (ingest backend, last directory, and scan
  limit) and
  `[library.ingest_options]` (per-type ingest options, persisted by the
  ingest canvas); `[library.search]` (recent-search history); `[notes]`
  (notes auto-save and sync); `[file_notes]` (File Notes root folder);
  `[rag]`, `[rag_search]`, and `[embedding_config]` for retrieval and
  embeddings.
- Child pages: [Media & conversations](library/media-and-conversations.md) · [Notes](library/notes.md) · [File Notes](library/file-notes.md) · [Prompts](library/prompts.md) · [Skills](library/skills.md) · [Collections](library/collections.md) · [Search & RAG](library/search-and-rag.md) · [Import & export](library/import-and-export.md)
- Deep dives: [Notes bidirectional sync](../Features/notes_bidirectional_sync.md) · [Transcription](../Features/TRANSCRIPTION.md) (audio/video ingest backends).

## Quirks & troubleshooting

- **A rail count shows "(N+)".** The count was sampled rather than fully
  tallied — there are at least N items; open the row for the real list.
- **Export is greyed out.** In server mode the Export row is disabled:
  "Export packages local content only." Switch to a local runtime to
  export a bundle.
- **Pressing "u" does nothing.** The shortcut only works while the
  Search / RAG row is selected — select it (or use the **Search
  Library…** box) first.
- **Clicking Study decks / Flashcards / Quizzes doesn't open Study.**
  That's by design — the row opens a Library-local staging canvas first
  ("see what carries over"); press **Continue in Study** inside it to
  actually leave Library, or **Escape** to return to the hub. Generation
  and review run in the Study screen; Escape there returns to this
  staging canvas.
- **The palette found "Notes" but opened Library.** The standalone
  Notes, Prompts, Skills, Ingest, Research, and Media screens were
  retired; their names now route to the matching Library row.

—
*Verified against dev @ f0379c035 — 2026-08-07 (TASK-2850: Notes ▸ Folder files
mode stays inside the Library rail/canvas frame; Escape returns to
Library notes; TASK-2851: the legacy Media Library screen is retired — "Media &
Content: Open Media Library" now deep-links into Library's Media row;
TASK-2854: the Study/Flashcards/Quizzes hand-off rows read "opens staging
canvas", not "opens Study"; the Study screen names itself "Library ▸
Study" and no longer boxes the Library nav tab; Escape returns from Study
to the Study decks staging canvas; TASK-2857: the rail/canvas/toast CTA is
"Import…" everywhere (was "Add content…"), the Export canvas/button reads
"Export bundle (.zip)" (was "Export chatbook"), and the full media
viewer's escape hatch reads "Open in Library ▸ Media" (was "Open in Media
manager", stale since TASK-2851 retired that route); TASK-2856: entering
a Media/Notes/Prompts/Skills list now focuses its first row so ↑/↓/Enter
work immediately (previously nothing was focused there, on entry or on
return); Escape now moves focus from a list to the rail, and returns from
that list's viewer/editor to the list, both newly advertised in the
footer; TASK-2856 re-critique round 3: the media viewer's Edit/Delete/Edit
analysis sub-states now document their own graduated two-Escape behavior
and footer hint ("back a step") instead of implying a single Escape
reaches the list from any viewer sub-state)*
*Verified against dev @ 6b38a13b8 — 2026-08-07 (task-2858 Task 4: rail
glosses/counts follow one rule across visits (LIB-15); the search box
selects a stale query on click too, not just on a second "/" (LIB-17);
Conversations/Flashcards/Collections fall back to a short label instead
of a mid-word ellipsis at narrow widths (LIB-18)).*
*Verified against dev @ 642567627 — 2026-08-10 (task-4011: the Study
screen's breadcrumb/Escape now name the actual origin — the Library-origin
round trip described above re-driven live and unchanged; the Home-origin
variant reads "Home ▸ Study" and Escapes to Home).*
*Verified against dev @ 023a04a48 — 2026-08-07 (task-2860: the "F6 next
pane" footer hint above was previously true in description only — a
`AppFooterStatus` filter silently dropped the landing's own F6 hint and
substituted the generic global "F6 panes" text instead. The footer now
renders the screen's own copy, live-verified at 170 and 100 columns; at
80 columns the whole screen-hint cluster (not just F6) already yields to
the width ladder described above, unchanged by this fix).*
*Verified against dev @ 023a04a48 — 2026-08-07 (task-2859 UAT P3 polish
batch, live-verified at 170x50: the Conversations canvas now opens with a
"Conversations (N)" title matching every sibling, and its filter box
renders above the empty-state text instead of below; the Collections
canvas title reads "Collections (N)" (was "Library Collections"); clicking
the Details header's LABEL (not just its **▾**/**▸** chip) now opens/closes
it too; the export quality caption ("keeps a small preview image…" /
"shrinks media files…" / "copies full media files…") now matches whichever
option is actually selected, not always "original"; the ingest queue tally
reads "This queue: N done" instead of the self-contradicting "N done — in
queue"; DB sizes in the Details disclosure include their `-wal`/`-shm`
sidecars, and the number/unit pair ("144.0KB") no longer wraps across two
lines at the rail's narrow width).*
*Verified against dev @ 023a04a48 — 2026-08-07 (task-3020: Escape now
cancels an armed Media bulk-delete confirmation instead of moving focus to
the rail with it still showing, matching the media viewer's own confirm;
a partial bulk-delete failure now focuses the first still-checked row
instead of leaving nothing focused).*
*Verified against dev @ 642567627 — 2026-08-10 (task-4023 AC#1–#4,
re-critique RC-07/09/10, live-verified with ANSI contrast measurement:
disabled Library action buttons now render at or above the 3:1 legibility
floor (they measured 1.39:1–2.30:1 before) and carry a leading "○" marker
plus a reason tooltip, so colour is never the sole disabled cue; the
Details disclosure's DB sizes recompute on open (a grown 12.3MB prompts
DB previously kept reporting its old size across close/reopen); F1's
panel is titled for the surface it describes ("Library Shortcuts —
Media"), lists each key exactly once, includes "F6 next pane" on
Search/RAG (whose footer now spells it verbatim), and a second F1 closes
the panel.)*
*Verified against fix/settings-appearance-crash @ 57ad075de — 2026-08-10
(task-4023 AC#5–#7: one footer grammar everywhere — the Notes workflow's
run-on hints ("Ctrl+S Save · Esc Notes") became per-key pairs, and the
global cluster spells the pane key "F6 next pane" to match; value-cycle
buttons ("type: All", "sort: Newest", "quality: thumbnail", "mode:
Search") carry a trailing "⇄" with a tooltip listing the full cycle —
a trailing "▸/▾" is now always a section-header disclosure and a leading
"▸ " always the selected list row (Collections rows included); the Media
toolbar is a single horizontal row like its siblings; canvas list titles
render in full instead of the rail's 17-character cut; the landing line
reads "pick a section" (no "on the left" — at ≤100 columns the shell
shows one pane at a time); Escape now works on Export (back to the
canvas that opened it, or the hub), Collections (focus rail), and the
Study staging canvases (back to hub); the staging rows' second line
reads "see what carries over".)*
*Verified against feat/library-queue-batch @ 0662e09f5 — 2026-08-11
(task-14902: the value-cycle buttons converged on the Notes Sort chooser
pattern — pressing "type: All" / "sort: Newest" / "quality: thumbnail"
opens a one-row choice strip with a "✓" on the active option and a direct
pick (Escape cancels; the footer/F1 read "enter choose … / esc cancel"
while a strip is open), so the trailing "⇄" now appears only on the
surviving genuine two-option toggles, sitting between the two enumerated
options with "✓" on the active one ("mode: ✓ Search ⇄ RAG Answer", the
skill editor's switches); the prompt collection control — a chooser that
opens the collection manager — dropped the glyph outright.)*
*Verified against `feat/rag-p2a-instrument-renewal` at 0c34be595 —
2026-08-11 (TASK-15020 final review wave, doc-only: correcting the "Find
anything you've saved" step above to match B3's already-shipped
behavior — the Search / RAG canvas's per-source count follows Settings ▸
RAG's Default results, 15 on the shipped default profile, not a fixed 5;
no code changed here).*
*Verified against feat/workspace-create-modal @ 64a07a3d7 — 2026-08-17
(task-18704: **Create local workspace** now opens the shared "New
Workspace" dialog — the same one Console and Settings use — instead of
creating a zero-input workspace instantly; documented its name prefill,
optional validated folder bindings, Browse… picker, and default-on
"Switch to this workspace" checkbox).*
*Verified against feat/project-skills-import @ 964cb04df — 2026-08-18
(task-18705: a bound folder containing `.SKILLS/` now annotates its row
"— contains N project skill(s)" and a chained import prompt follows
Create).*
*Verified against codex/library-top-level-pagination @ 937dfa393 —
2026-08-20 (TASK-19022: compact Get started rail, truthful unresolved and
recovery states, remembered Explore/Back disclosure, permanent graduation,
deep-link and palette bypass, and keyboard/compositor UAT with the production
stylesheet at 100x30 and 170x48). Per user direction, repository-wide pytest
was not run; only modified/touched Library component and direct-owner gates are
claimed.*
