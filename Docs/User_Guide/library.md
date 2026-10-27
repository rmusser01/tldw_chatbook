# Library — Your content hub: sources, search, and imports.

## What this screen is for

Library is where everything the app knows about lives: media you've
imported, conversations from Console, notes, prompts, skills, and
Collections, registered Chatbooks, and reports — plus search and RAG over all of it, and the import/export
tools that move content in and out. Reach for it to add source material,
find something you saved, or hand a bundle of sources off to Console or
Study. This page is the orientation tour; the details live on these pages:

- [Media & conversations](library/media-and-conversations.md) — browse imported media (with the media viewer) and your Console conversations.
- [Notes](library/notes.md) — the notes list, editor, templates, reviewed import, and lasting folder sync.
- [File Notes](library/file-notes.md) — the folder-backed File Notes workspace and its Session Git panel.
- [Prompts](library/prompts.md) — saved prompts: list, editor, import, and Console insert.
- [Skills](library/skills.md) — skill packs: import, editing, and the trust/approval flow.
- [Collections](library/collections.md) — the Quick Capture reading list: saved web captures, highlights, and the legacy-records recovery path.
- [Search & RAG](library/search-and-rag.md) — the Library Search/RAG canvas, evidence, and the Console handoff.
- [Artifacts](artifacts.md) — registered Chatbooks, all reports, kept copies, and sharing.
- [Import & export](library/import-and-export.md) — the Import media flow and the Export bundle (.zip) canvas.

## Getting there

- Press **Ctrl+3** from anywhere, or click **⌃3 Library** in the nav bar.
- **Ctrl+P** → "Tab Navigation: Switch to Library" in the command palette.
- Coming back this way always opens the Library landing, even if you left
  from Search / RAG or another canvas; your Search / RAG draft is kept. A
  deep link (below) opens its own canvas instead.
- **Ctrl+6** opens **All artifacts**. The Artifacts rail also has Chatbooks and Reports; Reports defaults to All reports with an explicit Kept filter. An empty Reports view offers **Set up a daily brief…**, which discloses the recurring watchlist, feeds, cadence and billed provider · model and asks you to confirm before creating anything.
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
**Explore all tools**. It has no search box, so **F6** lands on **Import…**
(or on **Explore all tools** when Import is not offered). The Get started
canvas offers the same journey as three controls that unlock in order —
**Import a file**, **Find it**, and **Use it in Console**. A step you cannot
run yet stays pressable and says why and what to
do first ("Find it needs something to search — Import a file first."), on the
line under the three controls and again if you press it. **Use it in
Console** unlocks once a search result is selected. While Library checks
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
is silent: the rail growing is the evidence, nothing is added to the canvas,
and whatever you were reading or typing is left alone. (Graduation was
announced by the toast "Library tools are now available." — superseded by
task-32555, which dropped it: the words described the rail rather than what
the reader had just done.)

Compact presentation never blocks navigation. Deep links and command-palette
routes, including **Tab Navigation: Library — Skills**, can open a tool that is
not shown in the Get started rail. A profile with no stored Library choice,
including one whose `config.toml` was written before its first launch, shows
the full rail while its sources are checked, then settles on Get started if
every source is empty and on the full Library if any holds content. A profile created by this app records its Library
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
  the order does not imply that items were ranked against each other. These
  rows are reachable by keyboard, and Enter opens whichever one has focus —
  so the focused row carries the same left-edge bar every other focused row
  on this screen does, and the footer names what Enter will do there ("enter
  open notes", "enter open media"). Before task-32546 focus here was a
  change of background colour and nothing else, and the footer never
  mentioned the row at all.
- **Quick actions** are **Import…**, **New note**, then **Search**. They use the
  same guarded destinations as the rail.

"Compact" here means the **single-stage** layout below 64 columns, not
merely a narrow terminal: at 100 columns the rail and the landing canvas
still paint side by side, with the landing's counts line, **From your
Library** and **Quick actions** all present. Only
once the screen drops to one stage does the rail become the sole navigation
owner and the landing canvas go away. task-32066 settled that the landing
does not yield earlier than 64 columns.

At compact widths the landing canvas **stays** beside the rail — both panes
are kept, the rail narrows, and whatever you had focused (Continue, a recovery
action, a cached summary, a quick action) keeps focus straight through the
resize. Only Notes' own workflow routes fold to a single pane at those widths;
the landing does not. Switching rail rows or resizing never leaves the keyboard
dead: focus is moved only onto controls that are on screen, and if a control
under focus disappears, the next key you press goes to the Library screen (Tab
moves on, F1 opens help).

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
**‹ Library** (or **< Library** with ASCII glyphs) to return, or press
**Escape** — the footer names it as `esc back to Library`. Widening restores
the co-present layout and prior focus/scroll position when no newer action has
replaced it.

The destinations with a three-pane reader (Media, Conversations, Notes,
Prompts, Skills, Collections) close their Library pane at that width too, and
Escape there does the same thing — **unless the canvas you are on already owns
Escape**. A surface with its own exit keeps it: a reader with an item open goes
back to its list, an editor back to its list, an armed delete confirmation
cancels, and Notes keeps its own Escape throughout (Notes still returns to the
rail — through its own key, not this one). The footer never names a return the
key would not perform; at 60 columns it may not have room to name the surface's
own exit either, so treat Escape as "back one step" and read the chip when
there is one. Above 64 columns a closed Library pane is an ordinary collapsed
pane one grip away, and Escape keeps its usual step-back through the visible
panes.

At this width the footer has room for one canvas key, and the way back wins it:
`esc back to Library` replaces the `/` and `F6` hints rather than being dropped
after them. Both keys still work; widen the terminal to see them named again.

A pane with nothing open gives its columns to its sibling: on Prompts, Skills,
Collections and Conversations the list widens to fill the canvas while the
"Select something to open it here." pane holds only its own floor, and
choosing an item restores the reading split. The landing hub is capped at a
readable measure rather than stretched across the whole canvas.
- **Header line** — reads **Library | Local**, or **Library | Server:
  \<label\>** when a server runtime is configured.
- **Left rail**, top to bottom. A new empty profile first sees the compact
  Get started controls above; the complete rail below appears after Explore or
  graduation:
  - a **Navigation** heading with **Collapse** at the opposite edge. Collapse
    hides the rail without changing the selected destination, search query,
    section disclosures, or canvas. The slim **Nav** handle expands it again;
    it is keyboard-focusable and remains part of the **F6** pane cycle. The
    handle is five cells wide, so it spells its name **downwards** —
    `N`/`a`/`v` down the column, with the collapse arrows below it. The
    handle beside the list pane does the same with that pane's name
    (**Items**, **Prompts**, **Skills**, **Folder files**);
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
    title with its count, plus a dim plain-language gloss on some of the
    jargon rows (Media — "your files", Prompts — "reuse", Skills — "AI
    add-ons", Search / RAG — "find all"). A gloss renders whole or not at
    all, and never flickers on or off just because a count arrived. Two
    rows stay bare on purpose: **Conversations** and **Notes** are
    already plain words and carry no gloss. **Collections** does have one
    — "saved captures" — but it is the longest, and it needs 34 cells of
    row width where the rail's default gives it 33, so on an ordinary
    wide terminal that row reads bare. Widen the rail past the default
    (Settings offers up to 48 cells) and the gloss appears; so does the
    single-pane narrow stage below 64 columns, where the rail takes the
    whole terminal rather than a fraction of it. When a row's title and
    count together would otherwise be cut mid-word, a handful of rows
    (Conversations, Flashcards, Collections) fall back to a short label
    ("Chats", "Cards", "Captures") instead of an ellipsis, so no row
    label ever cuts off mid-word and the count always stays visible.
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
  and "ctrl+n new note" (the hub's own accelerators);
  the Search / RAG canvas adds "u use Library
  context in Console", "o open evidence", and an "enter" hint that names
  whatever Enter does on the control you are focused on;
  a Media/Notes/Prompts/Skills/Collections list adds "esc focus rail";
  that list's item viewer/editor (or the media viewer) adds "esc back to
  list" instead; the Export canvas adds "esc back to Media" (or whichever
  canvas opened it — "esc back to hub" from the rail); and a Study
  staging canvas adds "esc back to hub". Every hint is a per-key
  "key action" pair — the Notes editor, for example, shows
  "esc back to notes" and nothing else, because the note is already
  saved and there is no save key to advertise — beside it, whenever one of
  the editor's own controls rather than a text field has focus, the footer
  also names that control's Enter action ("enter save note"), in Edit and
  Preview alike. In the notes LIST the footer names Enter only for the
  delete receipt's **Undo** and **Dismiss**; a note row or the filter shows
  no "enter …" chip. See [Notes](library/notes.md).
  While a text field has focus the footer leads with `typing in field`,
  names the Escape chip for that surface (`esc leave field` where nothing
  else owns the key — a list canvas keeps its own `esc focus rail`), and
  carries the canvas verbs after `after esc:` — they are not live until you
  leave the field. A list canvas's own filter lists none: its Escape moves the
  caret into the rail's "Search Library…" box, another text field, so one
  Escape would not make them live. Under 64 columns there is room for a
  single chip, so the footer names only what Escape does there and the verbs
  are not listed.

One special case: selecting **Notes** adds a
**Library notes | Folder files** strip above the workbench. **Folder files**
swaps the canvas pane for the File Notes workspace. At 120 columns and wider
the rail stays beside it; on compact terminals the canvas becomes the single
visible stage so its controls remain on-screen. Escape (or the
**Library notes** link) returns to the notes list — see
[File Notes](library/file-notes.md).

Opening a **note for editing** is the one place the rail steps aside on a
wide terminal, deliberately: at 120 columns and more of editor width the
rail collapses to its **Nav** handle so the note gets the room, and Escape
(or **Nav**) brings it straight back. Expanding it by hand ends that for the
rest of the visit — the rail then stays put while you open one note after
another.

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
| `●` (inside a line) | not a state — it samples a colour | a highlight's colour swatch in the Media reader |
| `▸` / `▾` | disclosure | trailing on a section header, leading on a folder-tree node |
| `○` | a blocked or disabled action | any greyed action, always beside its reason or tooltip |
| `●` / `○` (in a radio group) | the one option chosen out of a set | Console's per-conversation **Library access** modal and the first-run setup wizard — the two radio groups that paint the pair |
| `✓` (leading, in a chooser) | the active value of a chooser | choice strips, kept toggles ("mode: ✓ Search ⇄ RAG Answer") |
| `▸ ` (leading, on a rail row) | the destination you are on | the left rail |
| `⇄` | press to switch between the two options either side of it | mode toggles |

A rail row never expands and a tree node is never a rail row, so the two
leading `▸` uses cannot collide on one control.

Radio groups are deliberately carved out of the `☐`/`☑` pair rather than
folded into it: a radio offers exactly one choice, and painting it as a
checkbox would promise a multi-select it does not have. In the two groups
named above, exactly one sibling is `●` and the rest are `○` — that is what
tells you it is a chooser — so a `○` there means "not this one", not
"blocked". Outside a radio group, `○` keeps its blocked/disabled meaning.
(A decision, not a law of nature: it can be revisited if radios and disabled
controls ever end up side by side often enough to confuse.)

**Not yet everywhere.** Every other radio group in the app — the Console
capture-policy, export and share dialogs, the Chatbook wizards — still uses
the stock control, which paints `●` on *every* option and tells the two
states apart by colour alone. Read those by which option is highlighted, not
by the glyph; bringing them onto the pair above is outstanding work.

### Left rail

| Control | What it does |
|---|---|
| **Collapse** | Hides the wide navigation rail in place and gives the canvas the reclaimed width. The choice lasts for the current Library screen session. |
| **Nav** | Expands a collapsed rail and returns focus to **Search Library…**. The handle spells `Nav` down its five-cell column, so it is readable without hovering. On compact terminals, Library's existing one-pane routing takes precedence and the manual collapse returns when the terminal is wide again. Used on the rail a note editor collapsed, it also ends that editor's claim on the width for the rest of the visit. |
| **Import…** | Opens the Import media canvas — see [Import & export](library/import-and-export.md). |
| **New note** | Opens the production note-creation canvas. It is shown directly in the Get started rail. |
| **Explore all tools** | Reveals and remembers the complete Library without changing section disclosures. |
| **Back to Get started** | Returns an explicitly expanded, still-empty Library to the Get started landing and compact rail, with focus on **Import…**. It is offered only after you have explicitly expanded a still-empty Library from Get started, and never after graduation. |
| **Search Library…** | Type a query and press Enter: lands on the Search / RAG canvas and runs it (empty submit just opens the canvas) — see [Search & RAG](library/search-and-rag.md). **x** beside the box empties it. The box shows the live query only on the Search / RAG canvas; every other canvas gets an empty box, and returning to Search / RAG restores the query and its results. Text typed into the box on another canvas and never submitted is discarded when you leave — it never becomes the Search / RAG query. |
| **Chunking Lab** / **Try selected text** | Under **Details ▸ Actions**, below the line "Chunking Lab — compare how text is split for search". Opens a full-screen A/B tool; **Escape** there returns to the Library canvas you came from, including from the sample editor. |
| **▾** / **▸** (section headers) | Open or collapse that rail section — see [State glyphs](#state-glyphs). |

### Browse rows

Media, Conversations, and Prompts replace an empty page's disabled paging and
selection controls with a useful next step. The exact total remains visible in
the title, but there is no meaningless “page 1 of 1” or “nothing to select”
mechanic.

The paging half of that rule is Library-wide: **Skills, Collections and the
media Trash** drop “page 1 of 1”, the boundary reason and both **Previous** /
**Next** controls too whenever everything fits on one page, keeping only the
item range. Every part comes back the moment a second page exists. Paging that
is *paused* rather than absent — a stale page whose totals are withheld — keeps
its controls, disabled and carrying the reason.

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

On an empty Media page the keyboard lands on that one button (**Import
media** or **Show all types**), or in the filter box when a typed query found
nothing. When an open item's page empties under the Reader and the canvas
returns to its list by itself, focus that was in the Media panes lands there
too; focus you had put elsewhere, such as the rail's search box, stays where
it is.

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
"DB sizes" lines from disk (sidecars included), so the numbers you see are
current as of that open, not a reading cached at some earlier repaint. Those
sizes sit inside Details' own **Diagnostics** disclosure, closed until you
open it (and closed again next time) — they are there when you want them and
out of the way when you do not.

| Group | Contents |
|---|---|
| **Status** | A "Source · Local" (or "Source · Server: \<label\>") line and a counts row ("Notes N · Media N · Conversations N"). |
| **Diagnostics** | A closed disclosure holding — once a reading exists — a "DB sizes" label with one line per database ("Prompts 180.0KB", "Chats/Notes 1.1MB", "Media 508.0KB"), a line each so no size is split across two rail lines. |
| **Workspace** | "Active · \<workspace name\>" and a "Handoff" line. With nothing blocked it is a bare count ("0 eligible"). When something is blocked it says what you cannot do yet, why, and the next step: "1 item can't be used in Console yet · not in this workspace · Link it from the conversation's header". That sentence is longer than the rail's 34-cell column, so it wraps — and its continuation lines are indented two cells under the first, which is how you can tell they belong to the row above rather than being rows of their own. Every Details row wraps that way, including the Status counts line. |
| **Actions** | The buttons below, plus the note "Everything here is stored on this machine · syncing to a server isn't available yet." |

| Action | What it does |
|---|---|
| **Create local workspace** | Opens the same "New Workspace" dialog Console and Settings use — a prefilled "Workspace N" name, optional folders to bind (validated as added, with a Browse… picker), and a "Switch to this workspace" checkbox (checked by default). Escape cancels with nothing created. Server sync and agent hand-off (ACP) aren't available yet. A bound folder containing a `.SKILLS/` project skills folder is annotated "— contains N project skill(s)" and, after Create, offers a chained import prompt — see [Project skills](library/skills.md#project-skills-skills). |
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
forward: …" line naming up to three source titles (then "and N more.",
where N counts the rest of the titles Study keeps — the Study screen's
own scope banner names and counts the same ones), the ownership note
"Generation and review run in Study.", a readiness line, and a **Continue
in Study** button ("Open \<X\> with the current Library source snapshot,
or globally when none is available."). The readiness line depends on your
runtime: in server mode it reads "Source snapshot is ready."; in local
mode it reads "Generating a pack from sources needs a tldw server; you can
still make cards by hand in Study.", because only a tldw server can build
a study pack from the snapshot (Study's dashboard says the same). With no
sources at all it prompts you to import sources or create notes first.

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
| Tab / Shift+Tab | Move to the next/previous control **within Library**. Once focus is in the rail or a canvas, Tab cycles there and never tabs off the end of a canvas into the top navigation bar, which is reached with its own keys (Ctrl+digit / F-keys). When Library opens — at startup, or on your first visit from another screen — focus starts on the navigation bar's first button (**⌃1 Home**, where Enter leaves Library), and Tab walks the bar's buttons before it reaches the rail. From there, on the landing, **/** or **F6** goes straight to the rail's **Search Library…** box, and Shift+Tab reaches Library's last control |
| Esc | Context-dependent — see below |

Entering a Media, Notes, Prompts, or Skills list (from the rail, or
returning from its item) focuses the list's first row, so ↑/↓/Enter work
immediately without tabbing to find it. Entering Conversations from the
rail focuses its first row too. Escape then reads the surface
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
  **Delete**, which puts "Delete N selected items? You can undo right
  away, or restore later from Trash." above the list's toolbar and swaps
  that toolbar for **Delete** / **Cancel**) — Escape cancels it in
  place, exactly like its own **Cancel** button, instead of moving focus
  to the rail; the footer chip reads "esc cancel delete" while it's
  armed.
  Confirming with **Delete** when only some items can be removed leaves
  the failed one(s) checked and focuses the first of them, rather than
  leaving nothing focused or landing on an item you never selected.
- **In an item's viewer or editor** (the media viewer; the Notes,
  Prompts, or Skills editor) — Escape does exactly what **‹ Back to
  list** does on that surface, which is not the same on every editor.
  Notes autosave, so a note edit is already saved when you leave it:
  Escape returns to the list at once, re-focusing its first row, the
  editor's footer chip is "esc back to notes" (plus the focused control's
  own "enter …" action, when focus is on a control rather than a field),
  and its status line
  reads "Next: Keep editing; changes save automatically." Only a
  *blocked* save holds a note open, and it says what to fix — see
  [Notes](library/notes.md). Nothing autosaves in the Prompts or Skills
  editor: while you have unsaved edits, leaving the editor (Back,
  Escape, another row, another screen) is blocked until you save or
  resolve the edit. Both say so when they refuse, and the Prompts
  editor's Escape chip names whichever blocker is in force rather than
  promising a return to the list — "esc save or discard first" while the
  edit is unsaved, "esc busy, try again" while one of its writes is still
  running — see [Prompts](library/prompts.md).
- **Editing, deleting, or re-analyzing inside the media viewer** — a first
  Escape only closes that one form and returns to the plain read-only
  viewer (matching that form's own **Cancel** button); a *second* Escape
  from there returns to the list. The footer's hint changes to "esc close"
  while one of these forms is open, so it never claims "back to list" a
  step early. The Edit metadata and Edit analysis forms keep unsaved text:
  while what you typed differs from the saved value, Escape (and Back,
  another row, a rail row, the nav bar, Ctrl+Q) refuses with "Save or
  Discard changes first", the chip reads "esc save or discard first", and
  Cancel reads **Discard changes** — the one way to drop the text (see
  [Media and conversations](library/media-and-conversations.md)).

Ctrl+S saves in every Library editor — notes (flushes the autosave now),
prompts, skills, Folder files, and the Media analysis and metadata forms; it
does nothing outside an editor. It acts on the editor that holds focus, so
with focus on a list, a filter or the rail beside an open editor it does
nothing; with nothing unsaved it says "Saved — no changes." on that
editor's status line; where the editor's Save is held it says why there, and
it never moves focus. The footer and F1 show `ctrl+s` (with the editor's
verb) only while the key would save: a note or Folder files draft with unsaved
changes, a prompt or skill whose Save button is showing, or an open Media
form. Escape is also bound inside the skill editor
(back to list) — see [Skills](library/skills.md). Escape also
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
