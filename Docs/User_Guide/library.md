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
- [Notes](library/notes.md) — the notes list, editor, templates, reviewed import, and lasting folder sync.
- [File Notes](library/file-notes.md) — the folder-backed File Notes workspace and its Session Git panel.
- [Prompts](library/prompts.md) — saved prompts: list, editor, import, and Console insert.
- [Skills](library/skills.md) — skill packs: import, editing, and the trust/approval flow.
- [Collections](library/collections.md) — the Quick Capture reading list: saved web captures, highlights, and the legacy-records recovery path.
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

> **You will probably never see this.** "New profile" here means a profile
> whose `config.toml` was created **in the same run** — the first launch
> that writes the file. If the config file already existed when the app
> started, Library opens the **full rail** even on a completely empty
> profile, and the Get started view below never appears. That includes the
> ordinary case of quitting during first-run setup and relaunching: setup
> writes the config, so the next launch is no longer "new". Verified live
> on an empty profile with a pre-written config: full rail, every count
> `(0)`, no Get started controls. Making a genuinely empty profile reach
> Get started regardless of when its config was written is tracked as
> task-32059.

A new profile starts with a compact rail: **Import…**, **New note**, and
**Explore all tools**. The Get started canvas offers the same journey as three
controls that unlock in order — **Import a file**, **Find it**, and **Use it in
Console**. A step you cannot run yet stays pressable and says why and what to
do first ("Find it needs something to search — Import a file first."), on the
line under the three controls and again if you press it. While Library checks
its sources, it says
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
Library; deleting that content later does not hide tools again. Graduation
announces itself once, as the toast "Library tools are now available." — the
rail growing is the durable evidence, so nothing is added to the canvas, and
whatever you were reading or typing is left alone.

Compact presentation never blocks navigation. Deep links and command-palette
routes, including **Tab Navigation: Library — Skills**, can open a tool that is
not shown in the Get started rail. Existing profiles without this preference
open the full Library. A profile created by this app records its Library
lifecycle when the profile is created, so Get started still appears if you
complete first-run setup, quit, and come back before ever opening Library.

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
- **When the sources themselves fail to load**, the landing shows one bordered
  callout in their place, reading what failed and why, with its own **Retry**
  beside the reason — the counts line stays empty rather than reporting zeros,
  and **Continue** is withheld, because it leads somewhere else rather than
  fixing this. Two cases read differently: sources that missed the 5-second
  deadline ("Library sources did not answer · waited 5 s") are amber, and
  returning to Library re-runs that read once by itself; a hard failure
  ("Library source services unavailable; retry Library later. · \<reason\>")
  is red and waits for you to press **Retry**. The reason is the failure's own
  words only for an operating-system or database error; anything else is
  named by the kind of failure it is ("the connection failed", "the database
  could not be read", or "an unexpected error" when it is none of those), so
  a private path — and the exception's own text — never reaches the screen.
  Pressing **Retry** against an unchanged failure still repaints — the
  message gains "· attempt 2", "· attempt 3", and so on, so a press is never
  silent even when the outcome repeats. A brand-new profile sees the same
  callout on its Get-started view when that first source read times out or
  fails — it paints above the Get-started content rather than hiding it, so
  a new profile is never left with silent empty counts and no way to retry.
- **From your Library** uses cached summaries in the fixed order **Database
  Notes → Media → Conversations**. Missing or unresolved sources are omitted;
  the order does not imply that items were ranked against each other.
- **Quick actions** are **Import…**, **New note**, then **Search**. They use the
  same guarded destinations as the rail.

"Compact" here means the **single-stage** layout below 64 columns, not
merely a narrow terminal: at 100 columns the rail and the landing canvas
still paint side by side, with the landing's counts line, **From your
Library** and **Quick actions** all present (verified live at 100x30). Only
once the screen drops to one stage does the rail become the sole navigation
owner and the landing canvas go away. task-32066 settled that the landing
does not yield earlier than 64 columns.

At compact widths the landing canvas **stays** beside the rail — both panes
are kept, the rail narrows, and whatever you had focused (Continue, a recovery
action, a cached summary, a quick action) keeps focus straight through the
resize. Only Notes' own workflow routes fold to a single pane at those widths;
the landing does not.

## Layout tour

```text
+----------------------+------------------------------------------+
| Library rail         | Canvas                                   |
| navigation and tools | landing, list, viewer, editor, or import |
+----------------------+------------------------------------------+
| context-sensitive keyboard hints and status                    |
+-----------------------------------------------------------------+
```

When the rail is beside content, its automatic width follows the existing
3:13 Library-to-canvas proportion plus five cells and stays between 29 and 39
cells when space allows. Destination item lists default to 50 cells. An
explicit custom preference can be 24–48 cells; ordinary two-pane views may
temporarily shrink it to preserve 40 content cells, while adaptive readers
may collapse or prioritize panes. These responsive changes never overwrite
the saved preference.

Below 64 columns an ordinary Library route shows one stage at a time: either
the rail or the canvas. Activating a rail destination opens its canvas; use
**‹ Library** (or **< Library** with ASCII glyphs) to return. Widening restores
the co-present layout and prior focus/scroll position when no newer action has
replaced it.

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
    two-step trip out of Library), so they group under their own section —
    one row each. That click opens a
    Library-local staging canvas showing what will carry into Study, not
    the Study screen itself, and the canvas says so ("This page shows what
    carries over"); **Continue in Study** inside that canvas is
    the click that actually leaves, and **Escape** returns to the hub. The
    selected row is marked **▸**, and the Flashcards row shows "due: N"
    instead of a plain count;
  - a **Details** section, collapsed by default (see below). Section
    headers toggle open (**▾**) and closed (**▸**). **Chunking Lab** and
    **Try selected text** live inside it — see the control table below.
    When the rail is too short to show everything at once, its last line
    reads "▾ scroll for more" (hover it for the keyboard route: **F6**
    moves focus into the rail). The line disappears as soon as the whole
    rail fits.
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

### State glyphs

Every Library canvas uses this one legend — each glyph means exactly one
thing, so nothing on the screen relies on colour alone:

| Glyph | Meaning | Where |
|---|---|---|
| `█` (leading) | the keyboard cursor | focused list rows, focused evidence cards, the chooser cursor |
| `☐` / `☑` | selection you toggle | select-mode rows, the Search / RAG **Sources** panel, Import type toggles |
| `✓` / `✗` / `–` | a settled outcome | Import queue rows, receipts (`–` is "never attempted") |
| `≡` | already in your Library (a duplicate the import matched) | Import queue rows |
| `⊘` | cancelled on purpose | Import queue rows |
| `●` (leading a queue row) | still working | Import queue rows (queued, parsing, writing) |
| `●` (inside a line) | not a state — it marks a count, or samples a colour | the blocked count on the Workspace ▸ Handoff row; a highlight's colour swatch in the Media reader |
| `▸` / `▾` | disclosure | trailing on a section header, leading on a folder-tree node |
| `○` | a blocked or disabled action | any greyed action, always beside its reason or tooltip |
| `✓` (leading, in a chooser) | the active value of a chooser | choice strips, kept toggles ("mode: ✓ Search ⇄ RAG Answer") |
| `▸ ` (leading, on a rail row) | the destination you are on | the left rail |
| `⇄` | press to switch between the two options either side of it | mode toggles |

A rail row never expands and a tree node is never a rail row, so the two
leading `▸` uses cannot collide on one control.

### Left rail

| Control | What it does |
|---|---|
| **Collapse** | Hides the wide navigation rail in place and gives the canvas the reclaimed width. The choice lasts for the current Library screen session. |
| **Nav** | Expands a manually collapsed rail and returns focus to **Search Library…**. On compact terminals, Library's existing one-pane routing takes precedence and the manual collapse returns when the terminal is wide again. |
| **Import…** | Opens the Import media canvas — see [Import & export](library/import-and-export.md). |
| **New note** | Opens the production note-creation canvas. It is shown directly in the Get started rail. |
| **Explore all tools** | Reveals and remembers the complete Library without changing section disclosures. |
| **Back to Get started** | Returns an explicitly expanded, still-empty Library to the Get started landing and compact rail, with focus on **Import…**. It is never offered after graduation. |
| **Search Library…** | Type a query and press Enter: lands on the Search / RAG canvas and runs it (empty submit just opens the canvas) — see [Search & RAG](library/search-and-rag.md). **x** beside the box empties it. The box shows the live query only on the Search / RAG canvas; every other canvas gets an empty box, and returning to Search / RAG restores the query and its results. Text typed into the box on another canvas and never submitted is discarded when you leave — it never becomes the Search / RAG query. |
| **Chunking Lab** / **Try selected text** | Under **Details ▸ Actions**, above the line "Chunking Lab — compare how text is split for search". Opens a full-screen A/B tool; **Escape** there returns to the Library canvas you came from, including from the sample editor. |
| **▾** / **▸** (section headers) | Open or collapse that rail section — see [State glyphs](#state-glyphs). |

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

#### Media Trash

Open **Trash** from Browse › Media to recover local deleted items. Trash shows
at most 20 items per page in a stable newest-first order. **Previous** and
**Next** reach every page; the range and total are exact while the page is
fresh. A submitted title search and the exact type chooser filter the complete
Trash collection before paging. During loading or after a failed refresh, the
last good page can remain visible, but paging and destructive actions stay
disabled until **Retry** restores authoritative results. Selection belongs only
to the current page and filter.

**Restore** removes the item from Trash and marks the retained normal Media page
stale; it does not insert or reorder the restored item without a fresh Media
read. **Delete forever** (key `x`; `r` restores) requires inline confirmation
and cannot be undone.
Back returns to the exact normal Media page, selected item, list scroll, and
control that opened Trash. Returning from the Media viewer likewise restores
the exact list scroll and finishes focus on the item row. These returns settle
after the current Items layout is measured, including when the Library rail or
Items pane is independently collapsed; collapsing the Library rail gives the
Items title and detail area the reclaimed width.

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
| **Status** | A "Source · Local" (or "Source · Server: \<label\>") line, a counts row ("Notes N · Media N · Conversations N"), and — once a reading exists — a "DB sizes" label with one line per database ("Prompts 180.0KB", "Chats/Notes 1.1MB", "Media 508.0KB"), a line each so no size is split across two rail lines. |
| **Workspace** | "Active · \<workspace name\>" and a "Handoff" line. With nothing blocked it is a bare count ("0 eligible"). When something is blocked it names the reason and the next step: "2 eligible · 1 blocked · not in this workspace · Link it from the conversation's header". |
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
| / | Focus the filter of the list you're on — the Media, Conversations, Prompts, or Notes canvas's own **Title/keyword…** / filter box — so the footer's "/ focus search" lands where you're looking. On the landing, or on a canvas `/` isn't wired to (Skills, Collections, Search / RAG, Study), it focuses the rail's **Search Library…** box instead — those canvases have their own filter/query inputs, but `/` does not route to them today. Never fires while a text field already has focus. Get started has no hidden search target; use **Explore all tools** or a direct route. |
| u | Use Library context in Console — only while the Search / RAG row is selected (the footer hint appears only there) |
| ↑ / ↓ | Inside a Media, Notes, Prompts, or Skills list — or the New note canvas's Blank note / template rows — move to the previous/next row (stops at the first/last row — it does not wrap) |
| Enter | Open the focused list row (same as clicking it) |
| Tab / Shift+Tab | Move to the next/previous control **within Library**. Tab stays on this screen; the top navigation bar is reached with its own keys (Ctrl+digit / F-keys), never by tabbing off the end of a canvas |
| Esc | Context-dependent — see below |

Entering a Media, Notes, Prompts, or Skills list (from the rail, or
returning from its item) focuses the list's first row, so ↑/↓/Enter work
immediately without tabbing to find it. Escape then reads the surface
you're on:

- **While a long operation is running** (a Folder files folder change, a
  skill import, an export bundle write) — Escape, the back cue and Ctrl+Q
  all still work. A wait that outlives about three seconds says so in its
  own status line ("… · still working · Cancel") and offers a **Cancel**
  beside it; what a wait can refuse is a *second* write of the same kind,
  never your way out.
- **In any search or filter box** — the rail's **Search Library…** box, a
  canvas's own filter, the Search / RAG query box — Escape hands focus to
  the first control on the canvas, so the next key you press is a canvas
  key rather than another character in the box (press Escape then `i` on
  the landing and Import opens). Nothing you typed is cleared, and the
  footer switches from "typing in field" to the canvas's own hints. Where
  a surface already gives Escape a job (an editor, Import, Export, an
  armed confirmation — the entries below), that job still wins; the box
  simply stops holding the key hostage.
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

*Verified against fix/media-crit7-qodo — 2026-09-08 (task-32046 + task-32085
AC#2/#4: `/` focuses the active list canvas's own filter — Media/Prompts share
the per-canvas route the Conversations and Notes canvases already had — instead
of the rail's global search two panes away; the fallback wording now names the
filterless tool canvases rather than over-claiming "any" canvas. Pinned at
235x52 and 100x30 in `test_slash_focuses_the_media_filter_not_the_rail_search`
and `test_slash_focuses_the_prompts_filter_not_the_rail_search`).*

*Verified against fix/library-crit8-keyboard — 2026-09-08 (task-32051:
Escape now leaves a focused search/filter box for the canvas, so the next
printable key is a canvas key; task-32052: Tab stays inside the Library
screen instead of walking into the top navigation bar. Pinned in
`Tests/UI/test_library_crit8_keyboard.py`.)*

## Related settings & docs

- `config.toml`: `[library]` (ingest backend, last directory, and scan
  limit) and
  `[library.ingest_options]` (per-type ingest options, persisted by the
  ingest canvas); `[library.search]` (recent-search history); `[notes]`
  (note editor behavior; lasting sync state is device-private); `[file_notes]` (File Notes root folder);
  `[rag]`, `[rag_search]`, and `[embedding_config]` for retrieval and
  embeddings.
- Child pages: [Media & conversations](library/media-and-conversations.md) · [Notes](library/notes.md) · [File Notes](library/file-notes.md) · [Prompts](library/prompts.md) · [Skills](library/skills.md) · [Collections](library/collections.md) · [Search & RAG](library/search-and-rag.md) · [Import & export](library/import-and-export.md)
- Deep dives: [Lasting Notes folder sync](../Features/notes_bidirectional_sync.md) · [Transcription](../Features/TRANSCRIPTION.md) (audio/video ingest backends).

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
  (it opens with "This page shows what carries over"); press **Continue in Study** inside it to
  actually leave Library, or **Escape** to return to the hub. Generation
  and review run in the Study screen; Escape there returns to this
  staging canvas.
- **The palette found "Notes" but opened Library.** The standalone
  Notes, Prompts, Skills, Ingest, Research, and Media screens were
  retired; their names now route to the matching Library row.
- **A browse row says the Library sources are unavailable.** The message
  now carries a **Retry** beside it: press it to re-run the same source
  read without leaving the row. A repeated failure repaints with its
  attempt number; a success replaces the message with the real list.
  Failures a retry cannot clear (a policy denial, a runtime with no
  source services) keep the plain sentence and no button.

—
*Verified against fix/media-crit7-selreason — 2026-09-08 (task-32045: with
nothing checked, Select mode's Export/Review/Delete row now carries its own
always-visible "Select items to enable." line — the same inline-reason
grammar the Analyze row already used — instead of dimming with only the "○"
marker and a hover-only tooltip. The line disappears the moment any row is
checked.)*

—
*Verified against fix/media-riders-m — 2026-09-07 (task-31943: a bulk delete
followed by Undo puts the rail's "Media N" back to the restored total without
leaving the screen; task-31948: the browse-row source-failure message carries
its own Retry, in the same callout grammar the landing hub uses. Both verified
live at 235x52 on a scratch profile — the count round trip against a seeded
media DB, the callout against a profile whose media DB path is a directory.)*

—
*Verified against fix/media-wave5-g — 2026-09-05 (task-31632: the Library
landing paints one recovery callout with its own Retry when the source
snapshot fails, withholds Continue while it shows, and re-runs a timed-out
read on return. Verified live for the hard failure at 235x52 and 100x30 with
a scratch profile whose media DB path is a directory; the deadline case is
covered by the app tests, which is the only way to force it reliably.
Qodo PR #2451 round: the same callout now also paints on a brand-new
profile's Get-started view, above the starter content rather than in place
of it; the shared reason mapper redacts filesystem/database paths out of
`OSError`/`sqlite3` text before it can reach this or the Media callout; and
the Media pager's Retry re-requests only the page or type-facet fence that
actually failed instead of always reloading the page.)*

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
*Verified against fix/media-crit6-focus — 2026-09-07 (task-31983: a focused
list row now carries a distinct left-edge cursor bar, so keyboard focus reads
apart from the selected/open row even when a row is both — the bar moves with
↑/↓ and never changes the selection. Applied uniformly to the Media,
Conversations, Notes, Notes-folder and Prompts row canvases.)*
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
"▸ " always the selected list row (Collections rows included; narrowed
by task-32235 — a leading "▸" on a folder-tree node is disclosure, see
[State glyphs](#state-glyphs)); the Media
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
*Verified against fix/task-21116-wave4 (dev @ 30c7e1fe9) — 2026-08-23
(TASK-21116, performance conversion — no workflow changes: opening a media
item, leaving the media viewer (Escape / "‹ Back to list", including the
edit/delete/analysis Cancel steps), opening a Search/RAG result, the
section "Export…" actions, and the Prompts/Skills inline "Import…" row now
update only the affected canvas region instead of rebuilding the whole
screen. One visible refinement: opening the Prompts/Skills Import… row
parks the caret in its path field, and Cancel returns focus to the Import…
button.)*

*Verified against fix/task-22207-perf (dev @ 983aa5878) — 2026-08-25
(TASK-22207, performance — no workflow changes: arrow-keying through Browse
Media's Items list no longer rebuilds the Reader's document body per
keystroke; the "Loading preview…" banner paints and clears in place, and
only the row you settle on renders its document, once. Behavior of the
settle delay, Read/Analysis/Highlights/Info modes, and Find is unchanged.)*

*Verified against feat/task-22500-reader-virtualization (dev @ 732105c2d) —
2026-08-26 (TASK-22500, performance — no workflow changes: the Reader's Raw
text view now paints only the rows in view instead of the whole document.
On a 2.5 MB document this cuts first paint from roughly 2.3 seconds to
under a millisecond of render cost, and every search keystroke or
Prev/Next match click from roughly 1.7 seconds to under a millisecond;
opening the reader still pays a one-time indexing cost proportional to
document size (under 150 ms even at 2.5 MB) whenever the pane is resized.
Scrolling, search highlighting, match navigation, and click-drag text
selection behave the same as before. The Rendered (Markdown) view was
measured but not changed by this task — very large documents opened in
Rendered mode remain slow to first paint; tracked separately in
TASK-22660.)*

*Verified against fix/task-23025 — 2026-08-28 (TASK-23025, performance — no
workflow changes: resizing the terminal and Tab/arrow focus moves no longer
re-walk the Library DOM on every frame, and the model-install progress line
is now built on first use instead of on every visit. The compact/emergency
width crossings, the Details disclosure, and install progress all look and
behave exactly as before.)*

*Verified against fix/media-wave4-d — 2026-09-04 (task-28007 AC#3/AC#4:
Media ▸ Select mode gained an **Analyze** bulk action on its own row under
Clear/Export/Review. It generates an analysis for every checked item in
one run, in list order, and reports progress in place ("Analyzing 3 of 40 ·
2 failed", then "✓ analyzed · 38 of 40 · 2 failed", or "✗ analyzed · 0 of 3
· 3 failed" when nothing succeeded) with **Retry failed** and **Dismiss**.
Items that already have an analysis are never overwritten silently: the
first press offers "N of M already analysed" with **Skip them** /
**Overwrite**. With no analysis provider configured the action renders
disabled as "○ Analyze" carrying the resolver's own reason as its tooltip;
a second press while a run is in flight says "Analysis already running";
and leaving Library mid-run stops it with a notice naming where it got to.
Details on the [media & conversations](library/media-and-conversations.md)
page.)*

*Verified against fix/media-wave4-d — 2026-09-04 (task-28007 AC#1/AC#2: an
import run left with analysis-skipped rows (no provider configured at the
time) can be fixed in one action once a provider IS configured — "Analyze
N skipped" above the import queue, over every skipped id currently in the
queue. Details on the
[import & export](library/import-and-export.md) page.)*

*Verified against fix/library-crit8-polish-shell — 2026-09-08 (task-32066:
this page claimed the landing hides below 120 columns; it does not, and that is
deliberate — "keep compact landing alongside rail" (1a6c293761) took it out of
compact single-stage on purpose and pinned the two-pane result at 80 and 100
columns together with a focus-stability contract. The page now describes what
the app does. Confirmed live at 100x30.)*

*Verified against fix/library-crit8-polish-shell — 2026-09-08 (task-32059:
Get started is no longer skipped for a profile that completed setup and
relaunched before its first Library visit — the lifecycle is recorded at
profile creation instead of being inferred from a missing key.)*

*Verified against fix/library-crit8-polish-shell — 2026-09-08 (task-32063:
"Library tools are now available." fires only when the compact Get started
rail actually gives way to the full one, not on a populated profile's first
source read. task-32064: the "Chunking Lab / Try selected text" strip left the
top of every canvas for Details ▸ Actions, with a one-line gloss, and Escape
in the Lab returns to the Library canvas it was opened from. task-32069: the
rail search box has an "x" and no longer carries a stale query onto another
canvas; the three Study rows are one row each. task-32072: Get started's
"1 Add · 2 Find · 3 Use" is now three live controls.)*

*Verified against fix/library-crit8-polish-shell — 2026-09-08 (fix round:
task-32063's graduation notice is now the toast and nothing else — the
in-canvas line that repeated it is gone, so one event has one surface and
nothing is added to the canvas a reader is working in.)*

*Verified against fix/library-crit8-polish-shell — 2026-09-08 (review of
PR #2531: Escape leaves the Chunking Lab from its focused sample editor too,
not only with focus outside a text field; Get started's **Use it in Console**
unlocks on a *selected* search result and otherwise says "run Find it and pick
one."; and the graduation toast also reaches a brand-new profile, whose
lifecycle goes straight from `unknown` to graduated without settling on
Starter.)*

*Verified against fix/library-crit8-docs — 2026-09-08 (task-32073, three
corrections from critique #8's docs-vs-live pass, each re-tested live at
235x52 and 100x30 on a seeded and an empty profile):
(1) **Get started** is reached only by a profile whose config.toml was
created in the same run — an empty profile with a pre-written config opens
the full rail with `(0)` on every row and never sees it (task-32059);
(2) the landing canvas is **not** hidden merely at narrow widths — at 100
columns it still paints beside the rail; only the below-64-column
single-stage layout drops it (task-32066);
(3) the **Chunking Lab / Try selected text** strip under the header, which
paints on every Library canvas, was undocumented (task-32064).)*

*Verified against fix/library-crit8-waits — 2026-09-08 (task-32055: Library's
structural waits report "still working · Cancel" past three seconds and never
gate Escape, the back cue, the palette or Quit).*

*Verified against fix/library-crit9-grammar — 2026-09-10 (task-32235: one
meaning per state glyph. "○" had carried three at once — a disabled action,
an unchecked source toggle and a settled "skipped" outcome; it keeps the
first (it is the only non-colour cue on tooltip-gated buttons) while
selection moves to "☐/☑" and the never-attempted outcome to "–". The
legend is now stated once, under **State glyphs**.
Fix round 1 adds the Import queue's own `●`/`≡`/`⊘`, and round 2 splits `●`
by context: leading a queue row it means "still working", inside a line it
is a count marker or a colour swatch, neither of them a state.)*

*Verified against fix/library-crit9-rail — 2026-09-10 (task-32219: the Layout
tour no longer claims the Chunking Lab strip sits under every canvas header —
the control table's "Details ▸ Actions" is now the page's only statement of
its placement, and the rail says "▾ scroll for more" when its content runs
past the fold. task-32212: the rail search row keeps the pane's frame at 235,
100 and 60 columns. task-32220: the rail heading ellipsises instead of
cutting to "Navigati". task-32226: rail text typed on another canvas and
never submitted no longer seeds the Search / RAG query box. task-32230: the
Details DB sizes take a line each, and the Handoff line names the blocked
item's reason and its next step.)*
