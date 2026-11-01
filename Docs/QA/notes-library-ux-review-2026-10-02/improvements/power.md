# Improvements: power-user and PKM lens (Library + Library ▸ Notes)

Target: origin/dev `2d34cbf80d`. These ideas go beyond fixing individual defects. Each one is meant to remove a **class** of findings from `master-findings.md` while moving Library toward the keyboard speed that PRODUCT.md sets as Chatbook's identity: "keyboard speed, information density, direct manipulation … predictable behavior".

**Inputs:** master findings (103), journeys j1–j7 (their improvement opportunities, task outcomes and emotional journeys), the four maps, `known-context.md` §4 (ADR constraints), PRODUCT.md and DESIGN.md.

**Live probe (this pass):** socket `nl-imp-power-1`, golden profile, 160x45. `Ctrl+P` followed by `Lisbon` shows **"No matches found"**. The profile holds the note "Trip planning: Lisbon" and the conversation "Trip planning: Lisbon in October". `Ctrl+P` followed by `note` lists only commands: "Quick Actions: New Note" and then `Switch to …` entries. Captures are in `improvements/power-evidence/01-palette-lisbon.{txt,ansi}` and `02-palette-note.txt`. The palette's providers (`app_command_providers.py:72-1032`) are command lists only, with no data provider.

**Code facts the ideas rest on (verified at HEAD):**
- Console already has a `Ctrl+K` switcher: `UI/Screens/chat_screen.py:1936`, `Widgets/Console/console_session_switcher_modal.py`. A smaller picker exists at `Widgets/Console/console_prompt_picker_modal.py` (373 lines).
- A note-link edge store already exists and every note writer maintains it. `note_links` is rewritten by `replace_note_links` (`DB/ChaChaNotes_DB.py:17976-17999`), and the backlink query is at `:17960-17970`. It recognises only the `(note://<id>)` tail (`_NOTE_LINK_TARGET_RE`, `:205`).
- Prefix and AND match builders exist (`Utils/fts5_match_forms.py:242` `build_prefix_match_expression`, `:348` `build_and_match_query`). Prompts and Personas already use prefix search. Notes keeps phrase-only matching by an explicit TASK-19558 choice (`:368-385`). `notes_fts` indexes only `title, content` (`DB/ChaChaNotes_DB.py:1097-1101`).
- Cross-page selection already exists in one place: the Prompts canvas shows "N selected · M on this page" (`Widgets/Library/library_prompts_canvas.py:759`).
- Saved searches already exist in one place: Collections (`Library/collections_capture_repository.py:767`, table `collection_capture_saved_searches`).
- User templates live in a JSON file (`Notes/template_store.py`; loaded by `Event_Handlers/notes_events.py:54-132`). They support `{date}` and `{time}` only, and nothing in the app can author one.
- Free single keys in Library today: `f a v q y # 1 2 3`, Backspace and `Ctrl+K`. None of them appear in `library_screen.py` BINDINGS (`:1010-1222`) or in the `on_key` accelerators (map-shell-nav §4.3).

## Summary

| # | Improvement | Effort | Impact | Findings it removes or shrinks | ADR touchpoints |
|---|---|---|---|---|---|
| 1 | **Ctrl+K "Go to…"**: one switcher for every Library item, plus search-or-create | M | 5 | N-21, N-09, N-04 (workaround), S-09, S-13, S-22, L-03 | 031 refinement; 067; 097; 104 |
| 2 | **One Library verb grammar**: a single table generates the bindings, footer, F1, palette and the guide | L | 5 | S-28, S-11, N-22, S-08, S-15, N-31, N-32, S-26, S-12, S-13, L-03, L-09, L-29 | 031 (rules 2, 3, footer truthfulness) |
| 3 | **Link-native notes**: `[[` completion, title-resolved edges, follow and back, Links / Unresolved | M | 5 | N-08, N-26, N-21, S-03 (partial) | 027; 073 / 021 (bytes); schema migration |
| 4 | **One query language** for every filter: prefix, any word order, `#tag`, `in:`, `-word`; live; keyword facets | M | 5 | N-09, L-15, L-13, L-24, S-04 (searchability), N-21 | 067 (bounded results) |
| 5 | **A selection that survives** paging and filtering, with real bulk verbs | L | 4 | N-14, L-19, S-06, S-28, N-23, N-29, task-32635 | 055 (one delete seam); 067 |
| 6 | **Capture sheet from anywhere**, with provenance | M | 4 | S-02 (capture half), S-03, S-04, N-13, L-34 | 031; 055 C; 104 |
| 7 | **Reading desk**: a companion note beside the reader, plus "Quote to note" | L | 4 | S-02, S-03, L-16, S-04 (partial) | **086 amendment** |
| 8 | **Saved views** (smart folders) on every destination's rail row | M | 3 | L-34, L-24, S-22 (partial) | 076; 067 |
| 9 | **Templates are notes**, plus "Today" | M | 3 | j3 task 6, j3 idea 5 | 027 |
| 10 | **Plain-Markdown vault export** and Session Git diff | M | 4 | N-20, L-04, N-11, L-18, N-36, S-04 | 021 / 029; 038 / 039 |
| 11 | **Vault-grade sync**: health on every surface, named roots, always an exit, structure kept | L | 4 | N-02, N-03, N-15–N-19, N-35 | 073; **059 amendment** (structure) |
| 12 | **Resume where you left off**: per-destination working context and history, across switches and relaunches | M | 3 | S-22, S-02 (state half), L-16, N-27 | 104; 086 |
| 13 | **Ctrl+Q-proof editors**: one save contract (draft journal, flush on quit, forgiving input) | M | 4 | N-01, N-06, N-07, N-25, N-33, L-08 | 027; 031 r2 (retire Skills Ctrl+S) |

Ideas 1–4 are the foundation. Idea 4's query parser feeds ideas 1, 5 and 8. Idea 1's title index feeds ideas 3, 6 and 9. Idea 2's verb table is where every key used by ideas 3 and 5–9 gets registered. Suggested order is at the end.

---

## Benchmarks: what to borrow, what not to

| Product | Borrow | Do not borrow (and why) |
|---|---|---|
| Obsidian | Ctrl+O quick switcher with recents first; the `[[` suggester; "update internal links" on rename; Outgoing / Linked / Unlinked mentions; search operators (`tag:`, `path:`, `-`); the core Templates and Daily notes plugins | Graph view (decoration; PRODUCT names "control-room theater" as an anti-reference). Free tiling workspaces and tabs (a layout tax in a terminal). Templater's JavaScript execution |
| Logseq | Daily journal as the capture target; creating a page by following an unresolved link | Block references and transclusion `((uuid))` (breaks plain-file round trips with sync). Datalog queries |
| nvALT / Notational Velocity | One box that is both search and create; incremental results while typing; no save button | A title-only flat list as the *only* organisation (Library already has folders and keywords) |
| Bear | `#tag` typed inline becomes structure; built-in "Untagged" and "Todo" smart sections; Markdown export with tags in front matter | Nested-tag hierarchy as a second folder system (folders already exist; two trees confuse) |
| Zettlr | Citations and a references section on export; a link-first editor | Pandoc and CSL configuration surfaced in the UI |
| zk / nb (terminal) | `zk new --template` with group config and date variables; `zk list --linked-by`; LSP-style link completion; git-backed plain files as the format | CLI subcommand grammar transplanted into a TUI (`nb 12 edit`); Chatbook should expose verbs, not argv |
| Zotero | Notes pane beside the PDF reader; saved searches; "select all, add to collection"; Better BibTeX's kept-up-to-date export | Its modal dialog sprawl (item edit, collection edit, preference panes) |
| Readwise / Reader | Highlight → note with a backlink to the exact passage; a notebook sidebar per document; reading position kept forever | Cloud-sync-first assumptions (Chatbook is local-first; PRODUCT principle 2) |
| DEVONthink | Smart groups; batch tagging; per-item "where is this used" | Its search syntax (NEAR, fuzzy, BEFORE/AFTER) and dense inspector stacks |
| Raycast | Root search over commands *and* data; an action panel on the highlighted hit; "create when nothing matches" | Global OS hotkeys (a terminal app cannot own them); third-party extension stores |
| lazygit / mutt / ranger | The same verbs in every panel; a `?` help screen generated from the live keymap | Vim-modal editing inside the note TextArea; multi-key leader sequences (`g i`), which Textual cannot bind and ADR-031 steers away from |

---

## 1. Ctrl+K "Go to…": one switcher for every Library item, plus search-or-create

**Problem.**
- Opening a known note takes `/`, the query, Enter, 8 Tabs and Enter: 11 keys plus the query (N-21; j3 task 2). Obsidian takes 2 keys plus the query.
- Prefix queries fail: "Lisb" returns 0 results (N-09).
- At ≥120 columns the tree cannot scroll, so the filter is "the only reliable way in" (N-04; j5 task table).
- The command palette knows commands but no items (live probe above).
- Reaching another canvas by keyboard costs 6–29 Tabs (S-09, S-13; j4 idea 4).
- There is no recents list and no "back" (N-21, S-22).

**Proposal (what the user sees and presses).**
- `Ctrl+K` anywhere in Library opens a centred modal titled **Go to…**. It is the same key and the same modal grammar as Console's session switcher.
- **Empty query** shows two lists:
  - **Recent**: the last 10 Library items opened, of any type. Each row reads `Trip planning: Lisbon · note · Unfiled · 2h`.
  - **Places**: Media, Notes, Conversations, Prompts, Skills, Collections, Search / RAG, Import, Recently deleted, Manage sync folders.
- **Typing** runs prefix-tolerant matching on titles, aliases and keywords across notes, media, conversations, prompts and skills.
  - Results are grouped by type with a text type label. Each type returns at most 8 rows, recency-boosted.
  - Idea 4's grammar works here too: `#thesis`, `in:Projects`.
  - `n:` `m:` `c:` `p:` `s:` narrow to one type.
  - `today` yields today's daily note (idea 9).
- **Enter** opens the hit through the existing deep links (`note_id`, `open_source_type=media`, …; map-shell-nav §5.5) **and moves the list highlight onto it**. That closes the L-03 class by construction: opening an item by id can no longer leave the cursor, and therefore the reader actions, on another row.
- **Down** moves into the results. On a highlighted hit:
  - `c` stages it in Console (appended, see idea 5);
  - `y` copies its link (`[[Title]]`, or a media link);
  - Enter opens it;
  - any other printable key goes back to the query.
- **No exact title match:** the last row reads `New note "Lisbon packing list"`. Enter creates the note in Inbox and opens the editor. This is nvALT's search-or-create, as an explicit row rather than an implicit side effect.
- **Ctrl+P:** a `LibraryItemsProvider` also lists the top 3 item hits, so `Ctrl+P Lisbon` finds the note. The palette stays command-first; item hits rank below exact command matches.

**User value.** Any item is `Ctrl+K` plus 3–5 letters plus Enter away, whatever the width, the tree's scroll or expansion state, or the canvas on screen. This replaces the tree as the primary way for power users to find things, and gives first-timers the "where did I put it" answer the filter does not.

**Effort M · Impact 5.** The modal pattern exists twice (start from `console_prompt_picker_modal.py`), the deep links exist, and title-prefix search is a call to `build_prefix_match_expression`.

**Precedent.** Obsidian quick switcher; nvALT; Raycast root search and action panel; VS Code `Ctrl+P`; Console's own `Ctrl+K`.
- *Don't borrow:* body full-text search inside the switcher. Keep the switcher to titles, aliases and keywords so it stays instant; body search stays in Search / RAG, one `Enter` away through a "Search all content for '…'" row.

**Risks and constraints.**
- ADR-031: `Ctrl+K` is neither reserved nor forbidden. Record a refinement: "`Ctrl+K` = Go to / Switch in every destination".
- ADR-067: queries are bounded per source (≤20), and each source answers its own top-N, so there is no generic Library data controller.
- ADR-097: lazy-import the modal, so boot cost is zero.
- ADR-104: return focus through the existing event-driven settle, never with timers.
- Duplicate titles (two "Reading list" notes in the seed) must show their folder.

---

## 2. One Library verb grammar: a single table generates the bindings, footer, F1, palette and the guide

**Problem.** The same key does different things, or nothing, from one list to the next:
- Media selects with `s`/Space; Notes selects with Enter and has no Space (S-28). `e` exports only in Notes. `c` exists only in Media and Conversations.
- Escape is a 15-step ladder whose meaning changes by destination (S-11). Escape in the Notes filter clears unsubmitted text (N-31).
- Arrow keys skip folder rows (N-22). F6 in the note body selects a line (N-32).
- The note editor has no key for Delete, Use in Console or switching mode (map-notes §4.1, "absent" row).
- Verbs hit the wrong item: Read later after opening by link (L-03); `c` on a conversation the filter hid (L-09); a Trash selection with a live item still in the reader (L-29).
- Letters fire while focus is briefly `None`: `i` jumps to Import mid-filter (S-08).
- The footer and F1 disagree and advertise inert keys (S-15). The User Guide is wrong on 8 of 12 key claims checked (S-26). The active mode is invisible (S-12). Arrival focus lands on the Nav grip (S-13).

**Proposal.**
- Adopt the grammar below as the Library contract. Implement it as one declarative `LibraryVerb` registry; each entry holds the key, the label, the per-surface predicate, the palette title and an F1 sentence.
- `check_action`, the footer chips, F1 and the palette entries (for example `Library: File into folder…  f`) are all **generated** from the registry.
- The User Guide's keyboard section is emitted as a derived artifact that `scripts/preflight.sh` checks for drift, the same way it already checks the CSS bundle.
- Three behavioural rules travel with the table:
  1. **Target rule.** A verb acts on the selection if there is one; otherwise it acts on the highlighted row.
     - If the item open in the reader is not in the visible list, verbs that would act on it are dimmed with "Not in the current list — clear the filter" (L-09).
     - Opening by id always moves the highlight (L-03).
  2. **Focus rule.** Single letters dispatch only while focus is on a row, a reader body or canvas chrome. They never dispatch while `screen.focused is None`, or within 300 ms of an Input submit (S-08).
  3. **Escape rule.** Field → list → rail, one level per press. Escape never clears typed text and never discards a draft, and it leaves select mode on every list (S-11, N-31).
- **Arrival:** focus lands on the list's first row (S-13).
- **Mode controls** show `✓ Edit` the way Search / RAG already shows `✓ Search` (S-12).

### The grammar (outside text fields unless noted)

| Key | Meaning everywhere in Library | Today | Change |
|---|---|---|---|
| `Ctrl+K` | Go to… (idea 1) | Console only | add |
| `/` | Filter *this* list, live (idea 4); Escape leaves the field without clearing it | Notes binding; Media / Prompts / Conversations via `on_key`; the rail search elsewhere | always the visible list's own filter |
| `↑ ↓` | Rows, **including folder rows and pagers** | folder rows skipped (`screen_constants.py:432-450`) | add the classes |
| `← →` | Collapse / go to parent · expand / first child | none | new |
| `Home End PgUp PgDn` | First / last / page | none | new |
| `Enter` | Open, or run the focused control | yes | — |
| `Esc` | Up one level (rule 3) | per-destination | unify; publish the ladder in F1 |
| `] [` | Next / previous item **without leaving the reader or editor** (the editor flushes first) | Media only | Notes, Conversations, Prompts, Skills |
| `Backspace` | Back along the item history (links, switcher jumps) | none | new (idea 3) |
| `1 2 3` | Reader modes, in a fixed order: Edit/Read · Preview/Analysis · Info | Tab loop (9 stops in the note editor) | new |
| `n` | New note. In a reader it opens a note **about this item** (idea 6) | Notes and landing only; inert in Media (S-02) | extend |
| `c` | Send to Console (Use in Console / Resume), **appending** | Media `c`, Conversations `c`, Search / RAG `u` | everywhere; `u` kept as a silent alias for one release |
| `s` · `Space` · `a` | Select mode · toggle row · select all *matching* | Media `s`/Space; Notes Enter | every list |
| `e` | Export the highlighted row or the selection | Notes select mode only | every list |
| `f` | File into a folder or collection… | none | Notes, Prompts |
| `#` | Keywords… (add or remove) | none | every keyword-bearing list |
| `t` | Move to Trash: confirm, then an Undo receipt (ADR-055 A) | Media reader only; a Notes delete takes 4 actions via Info ▸ Danger | every list and reader |
| `r` · `x` | Restore · Delete forever, in Trash views | Media `r`/`x`; Notes `r` only (N-29) | Notes Trash gains `x` |
| `y` | Copy a link to this item | none | new |
| `v` | Save the current view (idea 8) | none | new |
| `q` | Quote the selection into a note (idea 7) | none | new |
| `i` | Import | from any non-text focus, including mid-re-render | rule 2 only |
| `F1` | Help generated from this table, plus a 4–6 line "How <surface> works" | footer plus bindings (S-15) | generated |

Media's contextual `l`, `R` and `m` stay as they are. Notes' `g` (go to folder) stays. There is **no save key** (see idea 13): ADR-031 rule 2 forbids `Ctrl+S`.

**User value.** Learn one list and you know seven. The footer can be trusted because it is generated from the same predicate that gates the key. Verbs cannot silently act on an item the user is not looking at.

**Effort L · Impact 5.** Bindings and gates are spread through `library_screen.py` (`:1010-1222`, `:25253-25681`) and the footer sets (`:1243-1600`, `:4736-4831`). Consolidating them is the cost, but each later idea becomes cheaper.

**Precedent.**
- lazygit: the same verbs in every panel, and `?` is generated from the live keymap.
- mutt: the index and the pager share keys.
- Zotero: the same item verbs in every collection.
- VS Code's palette shows each command's binding.
- *Don't borrow:* vim modal editing in the note TextArea; leader sequences (`g m`, `g c`, proposed in j4 idea 4), because `Ctrl+K` with Places covers the jump; user-remappable keymaps in v1 (they make footer truthfulness unprovable).

**Risks.**
- Muscle memory for `u` and `o` in Search / RAG: keep them as aliases for one release.
- ADR-031 rule 3: the destructive keys `t` and `x` keep their confirmations.
- Plain digits do not collide with the `Ctrl+digit` destination keys.
- `#` is Textual's `number_sign`. Verify it in tmux and on macOS Terminal before shipping.

---

## 3. Link-native notes: `[[` completion, title-resolved edges, follow and back, Links / Unresolved

**Problem.**
- Hand-typed `[[Title]]` is dead text. Only `[[Title]](note://<uuid>)` counts, the UUID is shown nowhere, and Preview hands `note://` links to the OS URL handler (N-08).
- A new note's "Linked from" never resolves, or shows the previous note's backlinks (N-26).
- j1 task 5 (link and follow) failed. j3 called dead links "disqualifying for a Zettelkasten user".
- After following a backlink, the only way back is to search again (N-21).
- There is no link type from a note to a media item (S-03).

**Proposal.**
- **Author.**
  - Typing `[[` in the body opens a completion list under the caret. It draws on titles and aliases from idea 1's index, recents first.
  - `↑ ↓ Enter` inserts plain `[[Title]]`. A duplicated title inserts `[[Study/Reading list]]`. Typing `|` after a pick adds an alias. Escape leaves the typed text alone.
  - The last row, `Create "Someday project"`, makes an empty note and links it.
- **Resolve.**
  - Extend `extract_note_link_targets` (`ChaChaNotes_DB.py:208`) so `[[Title]]`, `[[Folder/Title]]`, `[[Title|alias]]` and `[[Title#Heading]]` resolve by unique, case-insensitive title. The existing `(note://id)` tail still wins where it is present.
  - Unresolved targets are stored by their text.
  - **The body is never rewritten on save.**
- **Rename.** Renaming a note that others link to asks once: `Update 3 notes that link to "Old title"? Update links · Leave as is`.
- **Follow.**
  - Preview is built with `open_links=False` and handles `LinkClicked` (`library_notes_canvas.py:2909-2913`). Note links open in the app and `media://` links open the reader.
  - In Preview, Tab cycles links and Enter follows.
  - In Edit, `Ctrl+]` follows the link under the caret (vim's tag jump).
- **Back.**
  - `Backspace` (outside a field) walks the history stack.
  - The back cue names the destination: `‹ Index — start here` instead of `‹ Notes` when the note was reached through a link.
- **See.**
  - Info shows `Links out (N) · Linked from (N) · Unresolved (N)`. Unresolved rows offer **Create note**.
  - `y` copies `[[This title]]`.
  - A new note starts at `Linked from (0)`, never "checking…" (N-26).

**User value.** Linking becomes a 3-key habit. Imported vault links and hand-typed links behave the same, and following a trail of notes is reversible without searching.

**Effort M · Impact 5.** The edge relation already exists and every writer maintains it (`replace_note_links`), and backlinks are already indexed. What is missing is authoring, resolution by title, and following. Storing unresolved targets needs a `target_title` column, which means a migration, a `VALID_TABLES` / index entry and a query-plan pin (CLAUDE.md gotcha 1).

**Precedent.** Obsidian's `[[` suggester and link updates on rename; Logseq page refs; Bear `[[` autocomplete; zk's LSP completion and `zk list --linked-by`.
- *Don't borrow:* graph view; block references and transclusion; silent rewriting of link text on save.

**Risks.**
- **Interop is the trap.** Today's canonical form `[[T]](note://id)` would reach synced Obsidian files as visible `(note://…)` text. Plain `[[Title]]` must therefore be the authored default; `note://` stays an import and disambiguation form only. This respects ADR-073 and ADR-021 byte preservation.
- Completion writes through the ADR-027 coordinator; it is not a second save path.
- Resolving by title needs a deterministic tie-break, so ambiguous titles stay unresolved and listed, never guessed.

---

## 4. One query language for every Library filter: prefix, any word order, `#tag`, `in:`, `-word`; live; keyword facets

**Problem.**
- The Notes filter is one quoted phrase over title and body:
  - `meet` → 0 results (`meeting` → 3);
  - `size chunk` → 0 (`chunk size` → 4);
  - keyword `zebra` → 0;
  - the provenance keyword `71d30fb1` → 0 (N-09, S-04).
- Five filter dialects behave differently: Media is as-you-type over titles, content and keywords; Notes, Conversations and Skills apply on Enter; Prompts is debounced; Folder files is live (j5 matrix).
- Reader Find needs Enter and shows the previous query's count (L-15).
- Search mode returns nothing for a question the document answers verbatim (L-13), and silently keeps the previous query's sources (L-24).

**Proposal.** A single `LibraryQuery` parser serves every list filter, Folder files search, `Ctrl+K` and saved views.
- **Grammar:**
  - words are ANDed in any order, and the last word is a prefix (`Lisb` → Lisbon);
  - `"exact phrase"`;
  - `-word` excludes;
  - `#keyword` (joins `note_keywords` and the media, conversation and prompt keyword tables);
  - `in:Projects/Thesis` (folder, or a prompt collection);
  - `title:`;
  - `type:pdf` (Media);
  - `edited:today | 7d`;
  - `is:untagged | unresolved | later`.
- **Live**, debounced as Media's filter is today. **Enter opens the first result**, which is the fast path N-21 asks for.
- A one-line hint while the filter has focus: `Any order · "phrase" · #tag · in:folder · -word`.
- **Each result says what matched:** `title`, `#homelab`, or a body snippet with the match **bold and underlined**, never colour alone.
- **Zero results suggest the nearest relaxation:** `No notes match "Lisb zebra" · 3 match "Lisb" · Search all of Library (Enter)`.
- **Scope is always on screen:** `2 results · Media only — Search all sources` (L-24).
- **Keyword facets.** An empty, focused filter shows the top keywords with counts as selectable rows (`#daily 80 · #thesis 12 · …`). The Notes tree gains a collapsed **Keywords** branch (j3 idea 3). The profile's 76 note-keyword links become navigation instead of decoration.

**User value.** Partial words, tags and fields all work the same in every list, so "find it again" stops depending on remembering an exact phrase.

**Effort M · Impact 5.**
- The match builders exist and Prompts and Personas already use prefix search.
- Notes stayed phrase-only by a deliberate TASK-19558 rule ("a seam that already bound a quoted phrase keeps it", `fts5_match_forms.py:368-385`). This idea is the *measured* behaviour change that docstring asks for, so ship it with a before/after result-count table on the golden profile.
- Keywords are joined, not added to FTS, so no FTS rebuild is needed.

**Precedent.** Obsidian search operators; Bear's `#tag`; nvALT incremental search; Gmail's `in:` and `-`; fzf's extended syntax; zk's `--match` and `--tag`.
- *Don't borrow:* regex by default; DEVONthink's NEAR / fuzzy / proximity operators; a boolean query-builder UI; Dataview or Logseq live-query blocks inside notes.

**Risks.**
- FTS5 injection: build only through the existing quoting helpers.
- Prefix queries without an FTS `prefix=` index scan term ranges. Measure on the 122-note golden and on a 10k synthetic profile before adding an index; an index would need a migration and a plan pin.

---

## 5. A selection that survives paging and filtering, with real bulk verbs

**Problem.**
- Notes select mode is export-only, "Select all N shown" miscounts, and selecting costs two keys per row. Re-filing or re-tagging 10 notes takes about 120 keys (N-14; j3 task 5).
- Media selection is cleared by any page turn, and there is no bulk keyword action (L-19).
- A second "Use in Console" replaces the first staged note (S-06).
- Folder pickers list only expanded folders (N-23). Trash cannot delete permanently (N-29). Bulk note delete is task-32635.

**Proposal.**
- **Selection model.** Each destination owns an **id set** that survives page turns, filter and sort changes, and rail switches within the session.
- **Chip:** `12 selected · 3 shown · Show only selected · Clear`. "Show only selected" is just the query `is:selected` (idea 4).
- **Keys** (idea 2):
  - `s` enters and leaves select mode;
  - `Space` toggles a row;
  - `Shift+↑/↓` extends a range;
  - `a` selects **all matching**, including unloaded pages: `Select all 37 matching "daily"`. The set is materialised from the query, so ADR-067 page bounds still hold for rendering.
- **Bulk verbs** (strip buttons and the same keys):
  - `f` **File into…**: a folder picker that lists *every* folder, with type-to-filter. The title names the subject: `Move 12 notes to…` (N-23).
  - `#` **Keywords…**: shows per-keyword coverage (`#daily on 10 of 12`) with Add / Remove.
  - `c` **Use in Console (12)**: **appends** to staged sources, de-duplicated by id, and the strip then reads `Staged · 4 sources` (S-06).
  - `e` **Export**.
  - `t` **Move to Trash** through the single ADR-055 seam, with one receipt: `✓ 12 notes in Recently deleted · Undo`.
  - In Trash, `x` **Delete forever**, with a confirm that says it cannot be undone (N-29).
  - Media keeps **Analyze** and **Read later**.
- **Confirms name what the user cannot see:** `Move 12 notes to Recently deleted? 9 aren't shown by the current filter.`

**User value.** Triage, re-filing and re-tagging become set operations instead of one-by-one chores. Context for Console can be built from several notes at once.

**Effort L · Impact 4.** The Prompts canvas already models cross-page selection (`library_prompts_canvas.py:759`), and the Media bulk delete with Undo receipt is the reference seam.

**Precedent.** Gmail's "select all conversations that match this search"; lazygit range select; Zotero's multi-select → Add to collection; DEVONthink batch tagging.
- *Don't borrow:* drag-and-drop moves (mouse-first); Finder-style Cmd-click as the only multi-select.

**Risks.**
- ADR-055: bulk and single deletes must share one seam and one receipt.
- ADR-067: no generic Library data controller, so each source owns its own set.
- Stale ids, where an item was deleted elsewhere, are dropped with a count: `2 selected items no longer exist`.

---

## 6. Capture sheet from anywhere, with provenance

**Problem.**
- Quick capture costs 7 keys plus 2 Tabs from Console. From Media it is impossible by key: `n` and `Ctrl+N` do nothing, and coming back loses the reading position (S-02; j3 task 1).
- The palette's "New Note" arrival shows a tree with blank rows (N-13).
- A copied passage carries no attribution (S-03).
- Console's "Capture as note" keeps no question and no source, only UUID keywords (S-04).
- Collections' Quick Capture is web-only and its note box is unlabeled (L-34).

**Proposal.**
- **One `Capture` sheet**, a modal about 8 rows tall that **never navigates away**. It is reachable from:
  - the palette (`Capture note…`, from any destination);
  - `n` in the Media, Conversations and Search / RAG readers, prefilled with `About: [paper-retrieval-practice](media://26)`;
  - `q` on a reader selection, prefilled with `> quote` and `— [title](media://26#chunk-12)`;
  - Console's **Capture as note**, prefilled with the answer plus a provenance block: question, provider · model, date, and `Sources:` with a link for each staged or cited item.
- **Fields.**
  - One text area: its first line becomes the title (nvALT / Drafts style), and inline `#tags` become keywords.
  - A destination row, `→ Inbox ▾`, remembered between uses.
  - `Append to: (new note) ▾`, which picks an existing note through idea 1's index; "today's note" is the first option (idea 9).
- **Keys.**
  - Escape **keeps** the capture and closes the sheet. An empty sheet is discarded silently (ADR-055 C).
  - An explicit **Discard** button removes the text.
  - Focus returns to the control the sheet was opened from.
  - Receipt: `Saved "Lisbon packing" to Inbox` with an **Open** button.

**User value.** Capture costs 1–3 keys from wherever the thought happens, it costs neither the reading position nor the editing context, and every captured passage or answer keeps a trail back to its source.

**Effort M · Impact 4.** It reuses the note-creation path, the existing Console capture action (`UI/Console_Modules/message.py`, `_capture_console_answer_as_note`), and idea 3's link forms.

**Precedent.** Notational Velocity / nvALT; Drafts' "append to"; Obsidian QuickAdd into an Inbox; Logseq's append-to-journal; nb's `nb add "…"`; Readwise highlight → note.
- *Don't borrow:* a global OS hotkey (not possible in a terminal); AI-generated titles; `Ctrl+S` to save (ADR-031 rule 2), because Escape-keeps is the safer model.

**Risks.**
- The `media://` link type needs a note → media relation (shared with idea 7). `media://` exists today only in `MCP/resources.py`.
- ADR-104: return focus through the event-driven path.

---

## 7. Reading desk: a companion note beside the reader, plus "Quote to note"

**Problem.**
- The reader and a note can never be visible together, and every rail switch discards the reading position and the open note (S-02). One read-and-note cycle costs about 13 inputs plus re-scrolling, and j7 task 2 failed.
- Returning to Media shows a row marked "loaded" beside an empty reader (L-16).
- Highlights are a dead end (`✕ Delete` only), and there is no way to quote with a source link (S-03).

**Proposal.**
- **Wide (work pane ≥140 cells):** `n` in a Media or Conversation reader opens a **Companion note** column, about 40% on the right.
  - It is the note bound to this item, created on first use and headed `About: paper-retrieval-practice`.
  - The reader keeps its scroll position. `F6` moves between reader and note.
- **Narrower:** `n` swaps reader ⇄ companion inside the same pane, and both keep their state, so pressing `n` again returns to the same paragraph.
- **Quote.** `q` on a reader selection, or on a highlight card, appends a block quote and `[title](media://26#chunk-12)` to the companion note.
  - Highlight cards gain **Send to note**.
  - A **Send 3 highlights** action works on a highlight selection.
- **Back-links.**
  - In Preview, the `media://` link opens the reader at that chunk.
  - Media Info lists `Notes about this item (2)`.

**User value.** Reading and note-taking become one place instead of two destinations, with zero context switches per cycle, and quotes are citation-grade the moment they are made.

**Effort L · Impact 4.**

**Precedent.** Zotero 7's notes pane in the reader; Readwise Reader's notebook sidebar; DEVONthink annotation files; Obsidian's linked panes (just the two-pane case).
- *Don't borrow:* free tiling or saved workspaces; drawing annotations on the PDF itself.

**Risks.**
- **This needs an ADR-086 amendment**, because the work pane is permanent and single-slot. Proposed amendment: an optional, destination-owned companion slot inside the work pane with its own grip. Rail and list still collapse first, and the companion collapses before the reader.
- The reader is already the narrowest pane in some layouts (L-25), so the 140-cell threshold must be measured on the work pane, not the shell.
- It shares the note ↔ media relation with ideas 3 and 6.

---

## 8. Saved views (smart folders) on every destination's rail row

**Problem.**
- Recurring filters must be retyped.
- Read later is split across three unconnected places and absent from Collections ▸ Reading (L-34).
- Search / RAG silently keeps the previous scope (L-24).
- A relaunch drops the filter (S-22).
- Only Collections can save a search today.

**Proposal.**
- `v` (or a **Save view** button beside the filter) saves the current query (idea 4), sort and scope under a name. The saved view appears as an indented row under its destination in the rail, for example `Notes ▸ Thesis to-dos`.
- **Built-in views** appear only while non-empty:
  - Notes: `Unresolved links (N)`, `Untagged (N)`, `Edited this week`;
  - Media: `Read later (N)` (fixes L-34's split) and `Not analysed`.
- Enter on a view row applies it. **The filter box shows the view's query**, so a view is inspectable and editable, never hidden magic.
- **Rename / Delete view** live in the row's More.

**User value.** Repeated triage ("my unread PDFs", "notes I never tagged") becomes one key instead of a retyped query, using the same grammar as everything else.

**Effort M · Impact 3.** Generalise Collections' `collection_capture_saved_searches` model (`collections_capture_repository.py:767`; the seed already shows "Study material" and "Unread favourites") to a per-destination store.

**Precedent.** DEVONthink smart groups; Zotero saved searches; Bear's built-in Untagged / Todo / Today; notmuch / mutt virtual mailboxes.
- *Don't borrow:* nested smart-folder hierarchies; live-query blocks inside notes.

**Risks.**
- ADR-076: views must not undo graduation or add onboarding. They appear only after the user saves one or a built-in becomes non-empty.
- Rail density at 33 cells: views sit inside each destination's existing disclosure.

---

## 9. Templates are notes, plus "Today"

**Problem.**
- There are 8 fixed templates and no user or vault templates (j3 task 6).
- User templates exist only as a JSON file the app cannot author (`Notes/template_store.py`, `Event_Handlers/notes_events.py:54-132`) and support only `{date}` and `{time}`.
- Import once skips the vault's `Templates/` folder (j3 idea 5).
- The seed's 80 hand-made "Daily log YYYY-MM-DD" notes show the daily-note habit with nothing to support it.

**Proposal.**
- **A `Templates` system folder** in Library notes, beside `Agent_Lessons`. Any note in it is a template.
  - Its first lines may set `folder: Journal/Daily`, `keywords: daily` and `title: Daily log {{date}}`.
  - The body may use `{{date}} {{time}} {{title}} {{source}} {{cursor}}`. Nothing else, and no code execution.
- **New note view** lists user templates first, then the shipped ones. An existing `note_templates.json` is offered once: `Move 3 templates into the Templates folder`.
- **Today.**
  - A **Today** button on the Notes toolbar, `Ctrl+K` → `today`, and the capture sheet's "Append to today's note" all open or create today's note.
  - "Today's note" is defined by the template with `daily: true`: the filled title, filed in its folder.
- **Vaults.** Import once and lasting sync recognise the vault's templates folder (Obsidian's `.obsidian/templates.json` setting) and offer it as the Templates folder.

**User value.** Repeatable notes (meetings, dailies, paper summaries) cost one action, land in the right folder with the right keywords, and are editable with the same editor as any note.

**Effort M · Impact 3.**

**Precedent.** Obsidian's core Templates and Daily notes; Logseq journals; `zk new --template` with group config; nb templates; Bear templates.
- *Don't borrow:* Templater-style scripting (JavaScript execution: a security surface with no local-first payoff); a journal as the default home screen.

**Risks.**
- The Templates folder must be excluded from search results by default (`in:Templates` brings them back).
- The template variables must not collide with Prompts' `{{notes}}` substitution (L-26).

---

## 10. Plain-Markdown vault export and Session Git diff

**Problem.**
- Exported front matter is invalid YAML for titles containing `:` or `*` (N-20).
- Bundles drop keywords (L-04).
- Note Export silently overwrites, starting at `~` (N-11).
- The .zip promises "full media files" but writes text with `media_N.txt` names (L-18).
- The only portable format is a Chatbook bundle that "is only useful to another Chatbook user" (j7 task 7).
- Session Git shows status but never a diff (N-36).

**Proposal.**
- **Export as Markdown folder**, next to the .zip:
  - one `.md` file per note, mirroring the folder tree;
  - front matter from a real YAML emitter: `title`, `tags` (from keywords), `created`, `modified`, `chatbook_id`, and `sources` (media and conversation references);
  - links written as `[[Title]]`, or optionally as relative `[Title](../Projects/Thesis/Method.md)`;
  - media citations collected into a `## References` section;
  - deterministic slug filenames with a collision suffix, so re-exporting into a git repository produces minimal diffs.
- **Re-exporting into a non-empty folder shows a plan first:** `4 changed · 2 new · 0 removed · Write`. No silent overwrite; the same plan guards single-note export (N-11).
- **After export:** `Open in Folder files` (edit in place) and `Review session changes` (Session Git).
- **Session Git gains View diff** for the selected row and for the staged set, reusing the conflict-compare unified-diff box (N-36).

**User value.** Notes leave Chatbook in a form Obsidian, Zettlr, pandoc and git understand, with tags and sources intact, and git users review exactly what changed before they commit.

**Effort M · Impact 4.**

**Precedent.** Bear's Markdown export with tags; Logseq's "Export graph as Markdown"; Zotero Better BibTeX's kept-up-to-date export; zk and nb (plain files in git as the format).
- *Don't borrow:* automatic push. ADR-038 and ADR-039 allow only a guarded, reviewed push of one exact commit.

**Risks.**
- Users may confuse a one-time export with sync. The copy must say "a one-time copy · later edits are not synced", and offer **Keep a folder synced** as the alternative.
- ADR-021 and ADR-029: Folder files treats disk as the only content authority, so opening the export there makes the files, not the Library notes, the copy being edited.

---

## 11. Vault-grade sync: health on every surface, named roots, always an exit, structure kept

**Problem.** The sync findings are the worst trust breaks in the review:
- one ordinary edit wedges a root permanently while the editor says "Saved" and the tree says "⇄ Sync managed · Ready" (N-02);
- a deleted synced note leaves the root at "✓ Up to date" (N-03);
- review rows for deleted or renamed files cannot be resolved (N-15);
- one non-UTF-8 file blocks the whole folder (N-16);
- Add from files gets stuck (N-17);
- every root is titled "Sync folder (name unavailable before cutover)" (N-18);
- "No writes yet." appears when the history was unreadable (N-19);
- nanosecond timestamps and exception class names appear in the copy (N-35).

j3 ended with "I would not point this at my real vault". j6: "would not point a real vault at Keep-synced".

**Proposal.**
- **Health on every surface where sync matters**, one vocabulary:
  - tree folder row: `⇄ Vault3 · in sync · 3s ago` or `⚠ Vault3 · paused — 1 file needs review`;
  - editor location line: `In Vault3 · written to disk 20:33` or `Not synced — review in Manage sync folders`;
  - the Library header carries a `⚠ 1 sync folder needs attention` chip while anything is blocked.
- **Roots named** from the folder basename plus its path. **Problems are per file**, with plain reasons (`skipped · not UTF-8 text`), never a whole-root refusal.
- **Always an exit:** `Pause and keep both copies` and **Disconnect** (today "not in this release"). Recovery copy never loops back to a button that fails.
- **Every note write in a managed folder emits a sync intent** (task-32633), so no path can leave "✓ Up to date" standing.
- **Structure-preserving sync:** vault subfolders appear as Library sub-folders under the root, instead of one flat managed folder (j3 idea 4).

**User value.** A vault owner can see at a glance, wherever they are, whether disk and Library agree, and can always back out without losing either side. That is the precondition for pointing a real vault at Chatbook.

**Effort L · Impact 4.**

**Precedent.** Syncthing's per-folder state plus "Failed items" with reasons; the Obsidian Sync status icon and sync log; git status.
- *Don't borrow:* last-write-wins or automatic merges. ADR-073 requires an explicit Keep file / Keep note / Keep both / Skip.

**Risks.**
- Structure preservation **needs an ADR-059 amendment**: today one lasting-sync root owns one derived membership, which is why synced notes cannot keep the vault tree (notes.md:1205-1209).
- Ship the honesty parts (health line, names, per-file reasons, exit) first; they need no ADR change.

---

## 12. Resume where you left off: per-destination working context and history, across switches and relaunches

**Problem.**
- Every rail switch discards the open note and the reader position (S-02).
- Returning to Media shows "loaded" beside an empty reader (L-16).
- A relaunch drops staged sources, the open note, the filter and the expanded folders, because `ScreenStateStore` is memory-only (S-22).
- "View 10 imported notes" lands with the folder collapsed (N-27).

**Proposal.**
- On every settle, each destination's working context is persisted per profile:
  - open item id, reader tab and scroll (chunk), editor caret line;
  - filter query, expanded folders, the selection set (idea 5), and the history stack (idea 3).
- Rail switches and relaunches restore it.
- **The landing's Continue** shows up to 3 rows: `Continue reading paper-retrieval-practice · 43%`, `Continue editing Thesis outline · 2h`.
- **Reveal-on-open:** any open-by-id (switcher, receipts, "View N imported notes") expands the folders above the item and scrolls it into view.

**User value.** Each daily session starts where the last one ended, and moving between Notes and Media stops costing re-navigation.

**Effort M · Impact 3.**

**Precedent.** Obsidian's workspace restore; Zotero reopening tabs at the same page; Readwise Reader's reading position; VS Code hot exit; browser session restore.
- *Don't borrow:* a tab bar of open documents. One open item per destination plus history is enough in a terminal.

**Risks.**
- ADR-104: restores must be event-driven and two-gate, never timer-based.
- ADR-086: geometry stays transient, and only *content* context is persisted.
- Restoring a note that was deleted elsewhere degrades to the list with `That note was deleted · Recently deleted`.

---

## 13. Ctrl+Q-proof editors: one save contract

**Problem.**
- `Ctrl+Q` quits without a flush or a prompt and loses unsaved note text (N-01, P0); B-static traces the same gap to prompt, skill and Folder files drafts.
- Opening any modal cancels the pending autosave (N-06).
- Validation vetoes yank focus mid-typing (N-07), and the leave-veto toast names a button that is not there (N-25).
- Four editors in one destination use four save models; `Ctrl+S` works only in Skills (N-33, a violation of ADR-031 rule 2).
- Escape in the analysis editor discards text without asking (L-08).

Power users type fast and quit fast, which is exactly the window these bugs live in.

**Proposal.** One contract for every Library editor (note, prompt, skill, Folder files, analysis):
1. **A draft journal inside the ADR-027 coordinator.** Every debounce tick writes the draft to a per-profile draft store, so a quit, crash or power loss loses at most a few hundred milliseconds of typing.
2. **Flush on quit.** `LibraryScreen.prepare_for_quit` awaits `flush_pending_work()`. The app's quit flow already asks every screen (ADR-031 refinement 33622.10). If the flush fails, the quit prompt names the item: `Quit and discard unsaved changes to "Ideas inbox"? Stay · Discard`.
3. **Recovery on next open:** `Recovered unsaved text from 20:11 — Keep · Discard`.
4. **Forgiving input, not vetoes.** Trim title whitespace and de-duplicate keywords on save, then say so (`Removed trailing space from the title`). Focus never moves.
5. **Modals never cancel autosave**; the debounce is re-armed after any pop.
6. **No save key.** Autosaving editors need none. Explicit-save editors (Prompts, Skills) answer Escape with the same `Save · Discard · Keep editing` prompt. Retire Skills' `Ctrl+S` (`library_screen.py:1047`) per ADR-031 rule 2.

**User value.** "Saved" becomes a promise that holds under the user's fastest habits, and the same mental model works in every editor.

**Effort M · Impact 4.**

**Precedent.** VS Code hot exit; vim swap files and recovery (a terminal-native precedent); Bear and Obsidian (no save button, always saved).
- *Don't borrow:* vim's dense swap-file prompt; Save As dialogs.

**Risks.**
- ADR-027: the journal must live inside the coordinator, **not** as a second save path.
- The draft store is user data: it is covered by backup, and drafts older than 30 days are pruned with a receipt.
- Folder files drafts must respect ADR-029 (disk authority): a recovered draft is offered, never written over a file that changed on disk.

---

## Sequencing (dependencies first)

1. **Trust floor:** 13 and the honesty half of 11 (health line, names, per-file reasons, exit). Nothing else lands well while quitting or syncing can lose work.
2. **Foundations:** 4 (the query parser) and 2 (the verb registry plus generated footer, F1 and guide).
3. **Speed:** 1 (switcher, using 4), then 3 (links, using 1's index), then 5 (selection, using 4 and 2).
4. **Flow:** 6 (capture), 12 (continuity), 8 (saved views, using 4), 9 (templates, using 1 and 6).
5. **Amendment-gated:** 7 (ADR-086), 10, and 11's structure preservation (ADR-059).

## ADR touchpoints

| ADR | Ideas | What changes |
|---|---|---|
| 031 (keys, footer truthfulness) | 1, 2, 13 | Refinement: `Ctrl+K` means Go to / Switch in every destination. The single-letter verb table becomes the source of truth for footer and F1. Retire the Skills `Ctrl+S` (rule 2) |
| 086 (adaptive reader shell) | 7, 12 | **Amend** for an optional companion slot in the work pane; persisted context is content-only |
| 059 (folder import / sync ownership) | 11 | **Amend** so a lasting-sync root can own a derived sub-tree |
| 055 (reversibility) | 5, 6, 13 | Bulk trash and purge reuse the single seam and receipt; an empty capture is silent GC (pattern C) |
| 067 (pagination) | 1, 4, 5, 8 | Bounded per-source result sets; "select all matching" materialises ids from the query, not from pages |
| 027 (note session coordinator) | 3, 9, 13 | Completion, templates and the draft journal go through the one coordinator |
| 073 / 021 / 029 (sync and disk authority) | 3, 10, 11 | Never rewrite link text in synced files; plain `[[Title]]` is the authored form |
| 104 (event-driven returns) | 1, 6, 12 | Every focus and scroll restore uses the settle gates, never timers |
| 097 (boot budgets) | 1, 6, 7 | New modals and panes are lazy-imported |
