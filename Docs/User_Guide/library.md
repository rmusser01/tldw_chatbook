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

## Layout tour

![Library overview](images/library/overview.svg)

- **Header line** — reads **Library | Local**, or **Library | Server:
  \<label\>** when a server runtime is configured.
- **Left rail**, top to bottom:
  - the **Import…** button ("Add files, links, and transcripts to
    your Library.");
  - the **Search Library…** box — submitting it lands on the
    Search / RAG canvas and runs your query (an empty submit just opens
    the canvas). Press **/** anywhere outside a text field to jump
    straight into this box; pressing **/** again inside it selects the
    whole query so the next keystroke replaces it;
  - four sections — **Browse** (Media, Conversations, Notes, Prompts,
    Skills, Collections, Search / RAG), **Create** (New note, New prompt,
    New skill), **Study** (Study decks, Flashcards, Quizzes), and
    **Import / Export** (Import…, Export). Each row is one line: the
    title with its count, plus a dim plain-language gloss on the jargon
    rows (e.g. "Search / RAG — find all"). On narrow terminals
    the gloss drops rather than truncating into fragments, and the title
    ellipsizes, so the count
    always stays visible. The three Study rows are hand-offs (they are a
    two-step trip out of Library), so they group under their own section
    and add a second "opens staging canvas" line — that click opens a
    Library-local staging canvas, not the Study screen itself; **Continue
    in Study** inside that canvas is the click that actually leaves. The
    selected row is marked **▸**, and the Flashcards row shows "due: N"
    instead of a plain count;
  - a **Details** section, collapsed by default (see below). Section
    headers toggle open (**▾**) and closed (**▸**).
- **Canvas** (the right pane) — there are no tabs here: the canvas swaps
  to match whichever rail row is selected. Before you pick one it shows
  the landing hub: per-source counts, quick actions (Import… /
  Search / New note, also reachable with **i** and **n**), and one
  clickable row per recent item that jumps straight into it, under
  the guidance line "Search everything, pick a section on the left, or
  add something new."
- **Footer** — shows the keys that work where you are: "/ focus search"
  and "F6 next pane" on every canvas; the landing adds "i import content"
  and "n new note" (single-letter accelerators for the hub actions);
  the Search / RAG canvas adds "u use Library
  context in Console", "enter select evidence", and "o open evidence";
  a Media/Notes/Prompts/Skills list adds "esc focus rail"; and that
  list's item viewer/editor (or the media viewer) adds "esc back to
  list" instead.

One special case: selecting **Notes** adds a **Database | Files** strip
above the workbench. **Files** swaps the canvas pane for the File Notes
workspace — the rail and the rest of the shell stay put, and Escape (or
the **Database** link) returns to the notes list — see
[File Notes](library/file-notes.md).

## Features & controls

### Left rail

| Control | What it does |
|---|---|
| **Import…** | Opens the Import media canvas — see [Import & export](library/import-and-export.md). |
| **Search Library…** | Type a query and press Enter: lands on the Search / RAG canvas and runs it (empty submit just opens the canvas) — see [Search & RAG](library/search-and-rag.md). |
| **▾** / **▸** (section headers) | Open or collapse that rail section. |

### Browse rows

| Row | Opens | Details on |
|---|---|---|
| **Media** | The media list and viewer. | [Media & conversations](library/media-and-conversations.md) |
| **Conversations** | Your Console conversations, with preview and "Open in Console". | [Media & conversations](library/media-and-conversations.md) |
| **Notes** | The notes list/editor, plus the Database \| Files source strip. | [Notes](library/notes.md) |
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

### Details

Collapsed by default; click the **Details** header to open it.

| Group | Contents |
|---|---|
| **Status** | A "Source · Local" (or "Source · Server: \<label\>") line, and a counts row: "Notes N · Media N · Conversations N". |
| **Workspace** | "Active · \<workspace name\>" and "Handoff · \<summary\>" lines. |
| **Actions** | The buttons below, plus the note "Server sync WIP · local only". |

| Action | What it does |
|---|---|
| **Create local workspace** | "Create a local-only workspace and make it active. Server sync and ACP handoff remain WIP." |
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
staging canvas above.

## Common tasks

1. **Find anything you've saved.** Type into the **Search Library…** box
   and press Enter — you land on the Search / RAG canvas with results
   grouped as "Evidence · top 5 per source". Narrow with the **Sources**
   scope toggles ([Search & RAG](library/search-and-rag.md)).
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
| / | Focus the **Search Library…** box, from anywhere on the screen (unless a text field already has focus) |
| u | Use Library context in Console — only while the Search / RAG row is selected (the footer hint appears only there) |
| ↑ / ↓ | Inside a Media, Notes, Prompts, or Skills list, move to the previous/next row (stops at the first/last row — it does not wrap) |
| Enter | Open the focused list row (same as clicking it) |
| Esc | Context-dependent — see below |

Entering a Media, Notes, Prompts, or Skills list (from the rail, or
returning from its item) focuses the list's first row, so ↑/↓/Enter work
immediately without tabbing to find it. Escape then reads the surface
you're on:

- **On the plain list** — Escape moves focus to the rail's **Search
  Library…** box (the same target `/` and F6 use); it never leaves the
  canvas or changes what's shown.
- **In an item's viewer or editor** (the media viewer; the Notes,
  Prompts, or Skills editor) — Escape returns to that list, re-focusing
  its first row, exactly like pressing **‹ Back to list**. A dirty note
  or prompt edit vetoes the exit the same way Back does.

Escape and Ctrl+S are also bound inside the skill editor specifically
(back to list / save) — see [Skills](library/skills.md). Escape also
returns Notes ▸ Files mode to the Database notes view, and is live
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
  ("opens staging canvas"); press **Continue in Study** inside it to
  actually leave Library. Generation and review run in the Study screen;
  **Escape** there returns to this staging canvas.
- **The palette found "Notes" but opened Library.** The standalone
  Notes, Prompts, Skills, Ingest, Research, and Media screens were
  retired; their names now route to the matching Library row.

—
*Verified against dev @ 4acb17a0b — 2026-08-07 (TASK-2850: Notes ▸ Files
mode stays inside the Library rail/canvas frame; Escape returns to
Database; TASK-2851: the legacy Media Library screen is retired — "Media &
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
footer)*
