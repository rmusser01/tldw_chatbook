# Library reading desk — design

Status: Draft for owner approval (TASK-34000.42)
Date: 2026-10-03
Origin: Library + Notes UX review 2026-10-02 (`qa/notes-library-ux-review-2026-10-02/`), findings S-02, S-03,
L-16, L-25; improvement proposals IA-04 / R-28 (`improvements/ia-loop.md`, `improvements/ranked.md`).
Amends (on acceptance): ADR-086 (Library adaptive reader shell), ADR-084 (Library media reader IA).
Builds on: TASK-34000.25 (reader + note state survive rail switches), TASK-34000.11 (in-app link routing),
TASK-34000.1 (autosave max-wait + quit flush), TASK-34000.10 (autosave veto never moves focus),
amended ADR-031 (Ctrl+S saves in every editor).

## 1. Problem

Reading and note-taking never share the screen. In the 2026-10-02 review the researcher journey (Priya),
the first-time Notes journey and the Notes power-user journey all failed "read and take notes together":

- Opening a note *replaces* the reader. One read → note → back cycle costs ~13 inputs (8 clicks + 5 keys)
  and discards both the reading position and a new note (S-02).
- There is no note action in the Media reader at all; `n` does nothing there.
- A quoted passage loses its source (S-03); kept AI answers lose their provenance (S-04).
- Back in Media the row says "loaded" beside an empty reader (L-16); the reader is the narrowest pane (L-25).

The product's core loop is "ingest or select sources, reason over them, preserve useful outputs"
(PRODUCT.md). The loop breaks at its first joint.

## 2. Goals and non-goals

Goals (v1):
1. From a document in the Media reader, one key (`n`) opens that document's note beside it.
2. Typing in the note and scrolling the document never cost either side its state.
3. One key (`q`) quotes a passage into the note with a link that opens that passage again.
4. The note ↔ source relationship is portable (travels with the note's text) and queryable
   ("Notes about this (N)", "Sources (N)").
5. Side by side at 120x36 and wider; a lossless stacked fallback below that.
6. Fully keyboard-operable, with visible buttons for mouse-first users; no colour-only meaning.

Non-goals (v1): Collections and Conversations readers (they adopt the same companion slot later);
server-authoritative Collections notes (ADR-113); provenance for kept AI answers (S-04 — reuses the
link format later); restoring the desk after an app relaunch (ADR-033; tracked by TASK-34000.41 / S-22);
a new server sync domain (ADR-105).

## 3. Owner rulings (2026-10-03)

| # | Question | Ruling |
|---|---|---|
| Q1 | Which readers in v1? | **Media first.** Collections/Conversations adopt the slot in follow-ups. |
| Q2 | Which note opens? | **Linked companion**: the document's own note, reopened next time; a note→source relationship is stored. |
| Q3 | Where does the link live? | **In the note text + a device-local derived index.** No new server domain (ADR-105 unchanged). |
| Q4 | Layout | **The note takes the Items slot**; the reader stays the work pane and keeps its width. |
| Q5 | What does `q` quote? | **The mouse selection if any, else a keyboard paragraph cursor** (`j`/`k`). |
| — | Sections 1-5 below | Approved in conversation 2026-10-03; this document awaits approval with the mockups. |

## 4. Behaviour

### 4.1 Opening the desk
- In the Media reader, `n` (and a visible **Note** button in the reader toolbar) swaps the Items list
  for the document's companion note. The document does not move; it keeps its width and scroll.
- Which note:
  - One or more notes link to this document → open the most recently edited. The note header reads
    `Note · Notes — <title> · 1 of 2 ▾`; the `▾` chooser lists every linked note (title · age).
  - None → create `Notes — <document title>` in a `Reading notes` folder (an ordinary note folder),
    whose body starts with `Source: [<title>](media://<media-uuid>)`, caret on the next line.
- `q` with the desk closed opens it the same way, then quotes (§4.3) — one key from reading to quoted.

### 4.2 Working in the desk
- `n` toggles focus note ↔ reader; F6 cycles panes as everywhere.
- While the note's text field has focus every printable key types; desk keys never fire there.
- Ctrl+S saves now (amended ADR-031). Autosave, the max-wait (TASK-34000.1), flush-on-leave and the
  quit guard come from the existing note session (ADR-027) unchanged.
- `]`/`[` move to the next/previous document; the note slot follows to that document's companion, or
  shows `No note yet · n to start one` — nothing is created until `n`.

### 4.3 Quoting
- In the reader, `q` inserts at the note's caret, without moving focus:
  - the mouse selection when one exists in the reader; otherwise
  - the paragraph under the **paragraph cursor** — a left bar on the current paragraph, moved with
    `j`/`k`, shown only while the desk is open, and named in the reader status line (`¶12`).
- Inserted text:
  ```markdown
  > Retrieval practice improved 7-day retention by 21 percentage points.
  > — [paper-retrieval-practice, ¶12](media://4b0e…-uuid#p12)
  ```
- Receipt in the note status line: `Quoted ¶12 into Notes — paper-retrieval-practice`.
- An item with no text (image, audio without transcript): Quote disabled with a reason,
  `○ Quote — no text in this item`.

### 4.4 Leaving and returning
- Escape ladder (ADR-031): note → reader; reader → close the desk (Items list returns, receipt
  `Saved to "Notes — <title>" · n to reopen`); then the existing Library ladder.
- A refused or failing save never closes silently: closing asks `Keep editing / Discard changes`
  (Keep focused); a refused autosave reports at the note without moving focus (TASK-34000.10).
- Leaving Library or switching rail destination keeps the desk for the session: returning restores the
  document, its scroll, the note, and the caret (on top of TASK-34000.25).

### 4.4a Companion header
- Row 1: `Note · <note title> · 1 of N ▾` and a **‹ Items** button that closes the desk (the reader's
  `‹ Back` is hidden while the desk is open — Escape and ‹ Items are the two ways out).
- Row 2: save state (`Saved 20:12 · ctrl+s save`, or the last receipt such as `Quoted ¶12 into …`).
- Row 3: the mode strip `Edit · Preview · Info` and **Save**. **Save never drops off**: when the
  companion has fewer than 46 content cells the strip compacts (shorter labels / active marker only)
  so Save stays visible — Ctrl+S performs the *visible* commit action (amended ADR-031), and a hidden
  Save is the N-05 failure again.

### 4.5 Discovery
- Reader footer adds `n note beside` and, while the desk is open, `q quote · j/k paragraph`.
- Reader toolbar gains **Note**; the paragraph status gains a **Quote** button.
- F1 in the desk explains the desk in one sentence, then lists exactly the keys live in the focused pane.

## 5. Data and provenance

### 5.1 Text is the truth
- Source line: `[<title>](media://<media-uuid>)`. Quote attribution:
  `[<title>, ¶<n>](media://<media-uuid>#p<n>)`.
- `Media.uuid` is `UNIQUE NOT NULL` and stable across devices, so links are portable. MCP's existing
  `media://<integer-id>` resources keep their meaning; the in-app resolver distinguishes a UUID from an
  integer id.
- Because the relationship lives in ordinary Markdown it survives Notes sync, export, lasting folder
  sync to `.md` files and Obsidian. No new server domain: ADR-105 is unchanged.

### 5.2 Self-healing anchors
- Paragraphs are the **blank-line-separated blocks of the item's stored text** (a heading counts as a
  block, so it can be ¶1). The same segmentation backs the Rendered and the Raw view, so `j`/`k`/`q`
  and anchors work on items that only have a Raw view (plain text, transcripts).
- `#p<n>` opens the document scrolled to paragraph `n`.
- If paragraph `n` no longer contains the quote's opening words (the document changed, e.g. re-import),
  the reader opens and runs Find on those words, saying `Paragraph moved — found by text`. If nothing
  matches: `Passage not found in the current version` — never a silent wrong passage.

### 5.3 Derived device-local index
- Table in the ChaChaNotes DB:
  `note_source_links(note_id TEXT, source_uuid TEXT, first_seen TEXT, last_seen TEXT, PRIMARY KEY(note_id, source_uuid))`
  plus an index on `source_uuid`.
- Derived: rebuilt from the note body on every save through the note session, inside the save
  transaction, off the UI thread; backfilled once by the migration (scan existing note bodies for
  `media://<uuid>`). Never synced, never exported — the text carries the truth, so the index is always
  rebuildable.
- Migration duties (CLAUDE.md gotcha 1): `_CURRENT_SCHEMA_VERSION` +1 with a migration, a
  `VALID_TABLES['chachanotes']` entry, an `EXPECTED_CHACHANOTES_INDEXES` entry, and a captured query plan
  with `sqlite_stat1` absent plus a `scripts/index_plan_pin_census.tsv` row.
- Index failure never blocks a save: Info shows `Sources: couldn't refresh — Rebuild`.

### 5.4 What the index powers
- Media Info: `Notes about this (N)` — each opens in the desk.
- Note Info: `Sources (N)` — each opens its document (in the desk when it is a Media item).
- The desk's `▾` chooser.
- A note is "about" a document whenever its body links to it — quoting a paper into any note counts.

### 5.5 Edge states
| State | What the user sees |
|---|---|
| Source in Trash | Link shows `In Trash · Restore` (ADR-055 reversibility seam) |
| Source not in this Library (other device, deleted forever) | `Not in this Library — <title>`; no destructive action |
| Companion note deleted | Dropped from the index on the next rebuild; `n` creates a new companion |
| Companion moved into a synced folder | Ordinary note: lasting-sync rules apply (TASK-34000.2/.4 semantics) |

## 6. Layout (ADR-086 / ADR-084 amendment)

### 6.1 The companion occupies the list region
- The desk swaps the destination list for the companion in the shell's existing **list region**:
  same border, same grip; the grip label reads `Note` while the desk is open.
- The reader remains the work pane.

### 6.2 Widths
- Companion floor 40 cells; share `clamp(round(0.40 × available), 40, 72)`; the reader keeps the rest
  (reader width priority, ADR-084). `available` = shell content width minus grips.
- Navigation collapses transiently while the desk is open (saved preference untouched); explicit reopen
  follows the resolver's existing priority.

| Terminal | Navigation | Companion | Reader | Arrangement |
|---|---|---|---|---|
| 235x52 | grip | 72 | 149 | side by side |
| 160x45 | grip | 58 | 88 | side by side |
| 120x36 | grip | 42 | 64 | side by side |
| 100x30 | grip | full (95) | full (95) | stacked faces |
| 80x24 | grip | full | full | stacked faces |
| 60x24 | emergency stage | full | full | stacked faces |

(Widths measured from the rendered mockups through the shell's own `sync_layout`; grips are 5 cells.)

### 6.3 Stacked faces
- When companion floor (40) + reader floor (48) + grips (~10) do not fit (~98 cells), the shell stacks
  the two as **faces** of the work region. A one-row header names both, active first:
  `Reading · paper-retrieval-practice ⇄ Note · Notes — paper… (saved 20:12)`.
- `n` flips faces; nothing closes; both scroll positions and the caret survive.
- In stacked mode the list-region grip is hidden and the inactive face in the header is a button that
  flips to it.

### 6.4 Amendment text (to be accepted with slice D3)
- **ADR-086** — add: "The list region may host a destination-owned *companion* in place of the
  destination list. A companion is active work: when the companion and the work pane cannot both meet
  their floors, the shell stacks them as faces of the work region and never collapses the companion.
  The grip names what the region currently holds. Companion visibility is transient session state."
- **ADR-084** — add: "With a companion open, the Reader keeps width priority beyond the companion's
  share (`clamp(0.40 × available, 40, 72)`)."

### 5.6 How links look while editing (owner decision pending)
The note editor today is a plain text area: the source line and quote attributions appear as raw
Markdown (`[paper-retrieval-practice](media://4b0e…-36-char-uuid)`), which wraps over two or three lines
in a 42-cell companion. Preview renders them as `paper-retrieval-practice ↗`. The mockups show the
`↗` form in Edit; that requires an editor capability that does not exist yet (concealing the link
target, revealing raw Markdown when the caret enters the link). See the approval questions.

## 7. Keyboard, errors, accessibility, performance
- Desk keys (`n`, `q`, `j`/`k`, `]`/`[`) bind through `check_action` and only act while the reader has
  focus; footer/F1 advertise exactly the live keys (ADR-031 rule 4). This closes the S-08 / N-31 class
  for the desk.
- Paragraph cursor = the app's row-focus `█` edge on the current paragraph (focus colour while the
  reader has focus, muted while the note has it) + `¶<n> of <N>` text in the reader status line with a
  **Quote** button (never colour alone).
- One focus style in both panes (S-14): the heavy focus-colour edge the note body uses today; the
  reader's current amber focus border is replaced by it while the desk is open. Stacked header names both faces in text.
- Lazy mount of the companion on first `n` (ADR-097); event-driven restore on return (ADR-104);
  paragraph segmentation computed once per rendered document and cached with the reader content.

## 8. Delivery slices

Each slice is one atomic PR with a regression test that fails on dev first and a live check in an
isolated profile.

| Slice | Content | Depends on |
|---|---|---|
| D1 Source links | In-app routing of `media://<uuid>[#p<n>]` (shares TASK-34000.11's router), Trash / Not-in-Library states, self-healing anchors | TASK-34000.11 |
| D2 Derived index | Migration, `VALID_TABLES`, index + plan pin, rebuild-on-save, backfill, Info lines | — |
| D3 Shell companion | List region hosts a companion; width rule; stacked faces; grip label; ADR-086/084 amendments accepted | — |
| D4 Desk in Media | `n` open/create/toggle, `▾` chooser, `]`/`[`, Escape ladder, ‹ Items, navigation survival, footer/F1, Note button | D2, D3, TASK-34000.25, TASK-34000.39 (L-25 one-row byline) |
| D5 Quoting | Paragraph cursor, selection path, `q`, attribution, receipt | D1, D4 |
| D6 Loop UAT + docs | Live rerun of the researcher journey at 120x36 / 160x45 / 235x52; User Guide (Media, Notes); ADR status | D1-D5 |

Tests: real-app Pilot tests per behaviour; **rendered-geometry assertions** (actual `region.width`, not
CSS intent) at 235/160/120/100/80/60 columns for D3; in-memory SQLite tests for D2 (rebuild after
dropping the index, backfill of existing links, plan pinned without `sqlite_stat1`); resolver unit tests
for D1 (UUID vs integer, moved paragraph, missing source, trashed source).

## 9. Success measures
- Read → note → back: ~13 inputs → **1** (`n`).
- Quote with a working source link: **1** key.
- Reader scroll and note caret survive every switch (desk toggle, `]`/`[`, rail switch, leaving Library).
- Review journey j7 task 2 ("read and take notes together"): **fail → success** in the D6 rerun.

## 10. Mockups (for approval)
Rendered with a throwaway Textual mockup using the app's real stylesheet and Library chrome:
235x52, 160x45, 120x36 (side by side); 100x30 and 60x24 (stacked faces); plus the empty-note state
(`No note yet · n to start one`), the `▾` chooser open, the paragraph cursor with a quote just
inserted, and a link in the `In Trash · Restore` state. Location:
`Docs/superpowers/specs/assets/2026-10-03-library-reading-desk/`.

## 11. Risks
- `library_screen.py` is ~36k lines under a module-size ratchet: the desk lives in its own controller
  and widget modules; the screen only wires them.
- Paragraph segmentation is defined on the stored text (§5.2), not on the rendered widget tree, so the
  Rendered and Raw views can never disagree about `¶n`.
- The local `file://` byline (L-25) takes 3–4 reader rows at 120x36 in today's reader; TASK-34000.39's
  one-row byline must land before D4.
- The derived index adds a write to every note save; it shares the save transaction and is measured
  against the existing save-latency budget before D2 lands.
